"""bedesigner.py – Design base editing guides for specific genomic variants.

Public API
----------
design_bes(...)        -> (sgrnas: pl.DataFrame, sam_df: pl.DataFrame)
output_sgrnas(...)
output_guides_sam(...)
"""

from __future__ import annotations

import copy
import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import pyfaidx
import pysam

import blast_guides
import dbsnp_sqlite3
import get_vep
import protein_variant
import shared


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel strings that are embedded in variant strings to signal errors.
# Using a frozenset enables fast membership tests (O(1)) and avoids repeating
# the same long list of strings in every check throughout the module.
_ERROR_CODES: frozenset[str] = frozenset({
    'no_input_gene_given',
    'no_input_transcript_given',
    'no_input_mutation_given',
    'input_transcript_not_found',
    'input_gene_not_found',
    'reference_not_amino_acid',
    'mutation_not_amino_acid',
    'input_position_not_numeric',
    'input_position_outside_ref_sequence',
    'putative_intron_spanning_codon',
    'no_suitable_snv_found',
    'no_suitable_mutation_found',
    'wrong_reference_amino_acid',
    'non_existent_input_rsID',
    'genomic_position_not_numeric',
    'genomic_coordinates_not_found',
    'variant_is_improperly_formatted',
})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PamConfig:
    """Precomputed PAM / editing-window search parameters.

    Instances are constructed once per run by :func:`_setup_pam_config` and
    then passed as a read-only context to all inner computation functions.
    This makes threading straightforward – every worker thread receives the
    same immutable config object.
    """
    pam_length: int
    pam_relevant: str
    pamlist_real_string: list[str]
    total_length: int
    length_edit_window: int
    bases_before_ew: int
    bases_after_ew: int
    bases_after_ew_with_pam: int
    bases_before_variant: int
    bases_after_variant_with_pam: int
    count_ignored_n_in_pam: int
    count_of_other_in_pam: int
    start_of_pam_search: int
    end_of_pam_search: int
    length_of_pam_search: int
    start_before_hit: int
    end_after_hit_with_pam: int
    end_after_hit_guide_only: int


@dataclass
class _EditabilityResult:
    """Outcome of checking whether a variant is editable by a base editor."""
    editable: bool
    rev_com: bool = False
    be_string: str = ""
    original_alt: bool = False
    target_base: str = ""
    target_seq_ref: str = ""
    target_seq: str = ""
    target_seq_ref_start: int = 0


@dataclass
class _GuideAnnotation:
    """CDS-level annotation collected for a single guide."""
    gene_symbols: list[str]
    transcript_symbols: list[str]
    exon_numbers: list[str]
    first_transcript_exons: list[str]
    last_transcript_exons: list[str]
    codonss: list[list[str]]
    codonss_edited: list[list[str]]
    aass: list[list[str]]
    aass_edited: list[list[str]]
    aa_positionss: list[list[str]]
    splice_sites_included: list[str]
    synonymouss: list[str]
    consequences: list[list[str]]


@dataclass
class _GuideRecord:
    """All data associated with one candidate guide sequence."""
    guide_with_pam: str
    guide: str
    pam: str
    guide_start: str
    guide_end: str
    guide_chrom: str
    var_with_by: str
    edit_window: str
    num_edits: str
    specific: str
    edit_window_plus: str
    num_edits_plus: str
    specific_plus: str
    safety_region: str
    num_edits_safety: str
    additional_in_safety: str
    edit_string: str
    edit_pos_string: str
    specificity: str
    distance_to_center_variant: str
    distance_to_center: str
    annotation: _GuideAnnotation


@dataclass
class _VariantEditorResult:
    """All results for one (variant, base-editor) combination.

    The attributes are intentionally structured so that the outer list (one
    entry per guide) can be trivially parallelised in the future – each
    ``_GuideRecord`` is self-contained.
    """
    variant: str
    variant_real: str
    editable: bool
    be_string: str
    original_alt: bool
    target_seq_ref: str
    target_seq_ref_match: str
    target_base_ref: str
    target_base: str
    strand: str
    guides: list[_GuideRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input preparation helpers
# ---------------------------------------------------------------------------

def _resolve_rsids(rsids: list[str], dbsnp_db: str) -> list[str]:
    """Query *dbsnp_db* for *rsids* and return resolved variant strings.

    Each returned string is of the form ``"<rsid>:<chrom_pos_ref_alt>"`` or
    ``"<rsid>:non_existent_input_rsID"`` when the rsID is not found.
    """
    logger.debug("Resolving %d rsID input(s) using dbSNP database %s", len(rsids), dbsnp_db)
    rsidvars = dbsnp_sqlite3.transform_locations(
        dbsnp_sqlite3.query_rsids(rsids, dbsnp_db)
    )
    rsids_df = pl.DataFrame({'rsid': rsids})
    rsidvars = rsidvars.join(rsids_df, on='rsid', how='right')
    rsidvars = rsidvars.with_columns(
        pl.col("variant").fill_null('non_existent_input_rsID')
    )
    rsidvars = rsidvars.with_columns(
        rsid_variant=pl.concat_str(
            [pl.col('rsid'), pl.col('variant')], separator=':'
        )
    )
    resolved = rsidvars['rsid_variant'].to_list()
    logger.debug("Resolved %d rsID input(s) to variant records", len(resolved))
    return resolved


def _resolve_tmuts(
    tmuts: list[str],
    cdss: pl.DataFrame,
    ref_genome: pyfaidx.Fasta,
) -> list[str]:
    """Resolve transcript-notation mutations to genomic variant strings.

    Each returned string is of the form ``"<tmut>:<chrom_pos_ref_alt>"``.
    """
    logger.debug("Resolving %d transcript mutation input(s)", len(tmuts))

    all_genes = set(cdss.get_column('gene_name'))
    all_transcripts = set(cdss.get_column('transcript_name'))
    
    resolved: list[str] = []
    for tmut in tmuts:
        parts = tmut.split("-")
        transcript = '-'.join(parts[:-1])
        mutation = parts[-1]
        tmutvar = protein_variant.get_variant_from_protein(
            transcript, 
            mutation, 
            cdss, 
            ref_genome, 
            True, 
            all_genes, 
            all_transcripts
        )
        for snp in tmutvar:
            resolved.append(f'{tmut}:{snp}')
    logger.debug(
        "Expanded transcript mutation input(s) into %d genomic variant record(s)",
        len(resolved),
    )
    return resolved


def _prepare_variant_list(
    input_variant: str | None,
    input_file: str | None,
    input_format: str | None,
    cdss: pl.DataFrame,
    ref_genome: pyfaidx.Fasta,
    dbsnp_db: str,
) -> list[str]:
    """Build the flat list of variant strings from CLI / file input.

    Handles rsID and transcript-mutation expansion in-place.
    """
    variant_list: list[str] = []

    logger.info(
        "Preparing variant input from %s",
        f'file {input_file}' if input_file else 'CLI argument',
    )

    if input_file:
        input_df = pl.read_csv(input_file, separator=',')
        input_columns = input_df.columns

        if not input_format:
            if all(c in input_columns for c in ['variant', 'chr', 'pos', 'alt']):
                raise ValueError(
                    'Input file contains columns "variant", "chr", "pos" and "alt"!\n'
                    'Please use either "variant" or "chr", "pos"(, "ref") and "alt" '
                    'or manually define --input-format!'
                )
            elif 'variant' in input_columns:
                input_format = 'variant'
            elif all(c in input_columns for c in ['chr', 'pos', 'alt']):
                input_format = 'vcf'
            else:
                raise ValueError(
                    'Input file does not contain the column "variant" or the columns '
                    '"chr", "pos"(, "ref") and "alt"!\n'
                    'Please provide either "variant" or "chr", "pos"(, "ref") and "alt"!'
                )

        logger.debug("Using input format %s for variant preparation", input_format)

        if input_format == "variant":
            if 'variant' not in input_columns:
                raise ValueError(
                    'Input file does not contain the column "variant"!\n'
                    'Please provide it if selecting --input-format variant!'
                )
            variant_list = input_df["variant"].to_list()

        elif input_format == "vcf":
            if not all(c in input_columns for c in ['chr', 'pos', 'alt']):
                raise ValueError(
                    'Input file does not contain the columns "chr", "pos"(, "ref") and "alt"!\n'
                    'Please provide them if selecting --input-format vcf!'
                )
            if "ref" in input_columns:
                variant_list = input_df.with_columns(
                    variant=pl.concat_str(
                        [pl.col("chr"), pl.col("pos"), pl.col("ref"), pl.col("alt")],
                        separator='_',
                    )
                )['variant'].to_list()
            else:
                variant_list = input_df.with_columns(
                    variant=pl.concat_str(
                        [pl.col("chr"), pl.col("pos"), pl.col("alt")],
                        separator='_',
                    )
                )['variant'].to_list()

    elif input_variant:
        variant_list = input_variant.replace(' ', '').split(",")

    # Separate rsIDs and transcript mutations, resolve them, then re-add.
    rsids = [v for v in variant_list if len(v.split("_")) == 1 and v.startswith('rs')]
    for rsid in rsids:
        variant_list.remove(rsid)
    if rsids:
        variant_list += _resolve_rsids(rsids, dbsnp_db)

    all_genes_transcripts = set(cdss.get_column('gene_name')) | set(cdss.get_column('transcript_name'))

    tmuts = [
        v for v in variant_list
        if len(v.split("_")) == 1 and '-'.join(v.split("-")[0:-1]) in all_genes_transcripts
    ]
    for tmut in tmuts:
        variant_list.remove(tmut)
    if tmuts:
        variant_list += _resolve_tmuts(tmuts, cdss, ref_genome)

    if rsids:
        logger.debug("Expanded %d rsID input(s)", len(rsids))
    if tmuts:
        logger.debug("Expanded %d transcript mutation input(s)", len(tmuts))
    logger.info("Prepared %d variant input(s) after expansion", len(variant_list))

    return variant_list


# ---------------------------------------------------------------------------
# PAM / edit-window setup
# ---------------------------------------------------------------------------

def _setup_pam_config(
    pam: str,
    guidelength: int,
    ew_start: int,
    ew_end: int,
    iupac_nt_code: dict[str, list[str]],
) -> _PamConfig:
    """Precompute all PAM and editing-window geometry constants."""
    pam_length = len(pam)
    pam_relevant = pam.lstrip('N')  # leading Ns are position-ignored

    pamlist = list(pam_relevant)
    pamlist_real = [iupac_nt_code.get(nt, [nt]) for nt in pamlist]
    pamlist_real_string = [''.join(combo) for combo in itertools.product(*pamlist_real)]

    total_length = guidelength + pam_length
    length_edit_window = ew_end - (ew_start - 1)

    bases_before_ew = ew_start - 1
    bases_after_ew = guidelength - ew_end
    bases_after_ew_with_pam = total_length - ew_end

    # -1 because the variant itself must always land inside the editing window
    bases_before_variant = bases_before_ew + length_edit_window - 1
    bases_after_variant_with_pam = bases_after_ew_with_pam + length_edit_window - 1

    count_ignored_n_in_pam = pam_length - len(pam_relevant)
    count_of_other_in_pam = pam_length - count_ignored_n_in_pam

    start_of_pam_search = (
        bases_after_variant_with_pam - bases_after_ew - count_ignored_n_in_pam
    )
    end_of_pam_search = count_of_other_in_pam - 1
    length_of_pam_search = count_of_other_in_pam

    start_before_hit = guidelength + count_ignored_n_in_pam
    end_after_hit_with_pam = count_of_other_in_pam
    end_after_hit_guide_only = count_ignored_n_in_pam

    return _PamConfig(
        pam_length=pam_length,
        pam_relevant=pam_relevant,
        pamlist_real_string=pamlist_real_string,
        total_length=total_length,
        length_edit_window=length_edit_window,
        bases_before_ew=bases_before_ew,
        bases_after_ew=bases_after_ew,
        bases_after_ew_with_pam=bases_after_ew_with_pam,
        bases_before_variant=bases_before_variant,
        bases_after_variant_with_pam=bases_after_variant_with_pam,
        count_ignored_n_in_pam=count_ignored_n_in_pam,
        count_of_other_in_pam=count_of_other_in_pam,
        start_of_pam_search=start_of_pam_search,
        end_of_pam_search=end_of_pam_search,
        length_of_pam_search=length_of_pam_search,
        start_before_hit=start_before_hit,
        end_after_hit_with_pam=end_after_hit_with_pam,
        end_after_hit_guide_only=end_after_hit_guide_only,
    )


# ---------------------------------------------------------------------------
# Variant parsing helpers
# ---------------------------------------------------------------------------

def _parse_variant_coords(
    variant: str,
    ignorestring: str | None,
    all_genes_transcripts: set[str],
) -> tuple[str, Any, str, str, str]:
    """Parse a variant string into (chrom, position, ref, alt, variant).

    *position* is returned as a string; callers are responsible for
    converting it to ``int`` (after subtracting 1 for VCF→0-based).
    *variant* may have an error-code suffix appended when the string does
    not match any known format.
    """
    parts = variant.split("_")

    if len(parts) == 4 and parts[0].startswith('rs'):
        rsidchrom, position, ref, alt = parts
        _rsid, chrom = rsidchrom.split(":")
    # elif len(variant_coords) == 4 and len(variant_coords[0].split("-")) in [2, 3]: # does not work, since some genes have hyphen
    elif len(parts) == 4 and '-'.join(parts[0].split("-")[0:-1]) in all_genes_transcripts:
        tmutchrom, position, ref, alt = parts
        _tmut, chrom = tmutchrom.split(":")
    elif len(parts) == 4:
        chrom, position, ref, alt = parts
        alt = alt.upper()
        ref = ref.upper()
    elif len(parts) == 3:
        chrom, position, alt = parts
        alt = alt.upper()
        ref = ""
    else:
        variant = variant + ':variant_is_improperly_formatted'
        # Use the sentinel split as placeholder values so later code that
        # checks for error codes can safely detect this case.
        chrom, position, ref, alt = 'variant_is_improperly_formatted'.split("_")
        return chrom, position, ref, alt, variant

    if ignorestring:
        chrom = chrom.replace(ignorestring, "")

    return chrom, position, ref, alt, variant


def _get_target_base(
    ref_genome: pyfaidx.Fasta,
    chrom: str,
    position: int,
    variant: str,
) -> tuple[str, str]:
    """Look up the reference base at *position* on *chrom*.

    Returns ``(target_base_ref, variant)`` where *variant* may have a new
    error code appended if the coordinates are not found in the genome.
    """
    try:
        return str(ref_genome[chrom][position]), variant
    except Exception:
        # pyfaidx raises various exception types (KeyError, ValueError, etc.)
        # when a chromosome or coordinate is not present in the genome file.
        logger.debug(
            "Reference lookup failed for %s:%d",
            chrom,
            position + 1,
            exc_info=True,
        )
        variant = variant + ':genomic_coordinates_not_found'
        return 'genomic_coordinates_not_found', variant


def _variant_error_code(variant: str) -> str | None:
    """Return the error code embedded at the end of *variant*, or ``None``."""
    for code in _ERROR_CODES:
        if variant.endswith(code):
            return code
    return None


def _extract_variant_real(
    variant: str,
    error_code: str | None,
    target_base_ref: str,
    edits: dict[str, str],
    allpossible: bool,
    all_genes_transcripts: set[str],
) -> str:
    """Derive the 'real' variant string (stripped of rsID/tmut prefix).

    When *allpossible* is True the ALT field is replaced with the base-editor
    product so the output correctly reflects the intended edit.
    """
    rsidtmut_parts = variant.split(":")

    if error_code is not None:
        variant_real = rsidtmut_parts[1] if len(rsidtmut_parts) > 1 else variant
    # elif (
    #     len(rsidtmut_parts) == 2
    #     and (rsidtmut_parts[0].startswith('rs') or len(rsidtmut_parts[0].split("-")) in [2, 3])
    # ):
    elif (
        len(rsidtmut_parts) == 2 
        and (rsidtmut_parts[0].startswith('rs') 
             or '-'.join(rsidtmut_parts[0].split("-")[0:-1]) in all_genes_transcripts)
    ):
        variant_real = rsidtmut_parts[1]
    else:
        variant_real = variant

    if allpossible and target_base_ref not in _ERROR_CODES:
        parts = variant_real.split('_')
        if edits.get(target_base_ref, 'be_not_usable') != 'be_not_usable':
            parts[-1] = edits[target_base_ref]
        variant_real = '_'.join(parts)

    return variant_real


# ---------------------------------------------------------------------------
# Editability check
# ---------------------------------------------------------------------------

def _determine_editability(
    target_base_ref: str,
    alt: str,
    be_fwd: dict[str, str],
    be_rev: dict[str, str],
    be_name: str,
    allpossible: bool,
    pam_config: _PamConfig,
    position: int,
    ref_genome: pyfaidx.Fasta,
    chrom: str,
) -> _EditabilityResult | None:
    """Check whether *target_base_ref* / *alt* can be edited by this editor.

    Returns an :class:`_EditabilityResult` if the variant is editable, or
    ``None`` otherwise.
    """
    if (target_base_ref == be_fwd['REF'] and alt == be_fwd['ALT']) or \
       (target_base_ref == be_fwd['REF'] and allpossible):
        logger.debug(
            "Editor %s matches forward editability for reference base %s",
            be_name,
            target_base_ref,
        )
        target_seq_ref_start = position - pam_config.bases_before_variant
        target_seq_ref_end = position + pam_config.bases_after_variant_with_pam + 1
        target_seq_ref = str(
            ref_genome[chrom][target_seq_ref_start:target_seq_ref_end]
        )
        return _EditabilityResult(
            editable=True,
            rev_com=False,
            be_string=be_name,
            original_alt=(alt == be_fwd['ALT']),
            target_base=target_base_ref,
            target_seq_ref=target_seq_ref,
            target_seq=target_seq_ref,
            target_seq_ref_start=target_seq_ref_start,
        )

    if (target_base_ref == be_rev['REF'] and alt == be_rev['ALT']) or \
       (target_base_ref == be_rev['REF'] and allpossible):
        logger.debug(
            "Editor %s matches reverse editability for reference base %s",
            be_name,
            target_base_ref,
        )
        target_seq_ref_end = position - pam_config.bases_after_variant_with_pam
        target_seq_ref_start = position + pam_config.bases_before_variant + 1
        target_seq_ref = str(
            ref_genome[chrom][target_seq_ref_end:target_seq_ref_start]
        )
        return _EditabilityResult(
            editable=True,
            rev_com=True,
            be_string=be_name,
            original_alt=(alt == be_rev['ALT']),
            target_base=shared.revcom(target_base_ref),
            target_seq_ref=target_seq_ref,
            target_seq=shared.revcom(target_seq_ref),
            # For rev_com guides, target_seq_ref_start tracks the upper (3') genomic
            # bound of the sequence slice and is decremented each guide iteration.
            target_seq_ref_start=target_seq_ref_start,
        )

    logger.debug(
        "Editor %s cannot edit ref=%s alt=%s (allpossible=%s)",
        be_name,
        target_base_ref,
        alt,
        allpossible,
    )
    return None


# ---------------------------------------------------------------------------
# Codon / consequence annotation helpers
# ---------------------------------------------------------------------------

def _determine_consequence(
    aa: str,
    aa_edited: str,
    non_stop_aas: list[str],
) -> str:
    """Map an (aa, aa_edited) pair to a consequence label."""
    if aa in non_stop_aas and aa_edited in non_stop_aas:
        return 'MISSENSE' if aa != aa_edited else 'SYNONYMOUS'
    if aa != 'Stop' and aa_edited == 'Stop':
        return 'STOPGAIN'
    if aa == 'Stop' and aa_edited != 'Stop':
        return 'STOPLOST'
    if aa == 'Stop' and aa_edited == 'Stop':
        return 'SYNONYMOUS'
    if aa == 'StartM' and aa_edited != 'M':
        return 'STARTLOST'
    if aa == 'StartM' and aa_edited == 'M':
        return 'SYNONYMOUS'
    if aa == "codon_incomplete" and aa_edited == "codon_incomplete":
        return 'INDEFINITE'
    return 'COMPLEX'


def _lookup_adjacent_exon_base(
    ntpos: int,
    row: dict[str, Any],
    exon_number: int,
    gene_symbol: str,
    transcript_symbol: str,
    first_transcript_exon: int,
    last_transcript_exon: int,
    chrom: str,
    cdss: pl.DataFrame,
    ref_genome: pyfaidx.Fasta,
    direction: int,  # -1 for previous exon, +1 for next exon
) -> str:
    """Return the reference nucleotide for *ntpos* in an adjacent exon.

    Raises :class:`ValueError` when the adjacent exon cannot be found
    unambiguously (0 or > 1 match), so callers should wrap in try/except.
    """
    adjacent_exon_number = exon_number + direction
    adjacent_exon = cdss.filter(
        (pl.col('gene_name') == gene_symbol) &
        (pl.col('transcript_name') == transcript_symbol) &
        (pl.col('exon_number') == adjacent_exon_number) &
        (pl.col('first_transcript_exon') == first_transcript_exon) &
        (pl.col('last_transcript_exon') == last_transcript_exon) &
        (pl.col('Chromosome') == chrom)
    )
    if len(adjacent_exon) != 1:
        raise ValueError(
            f'Expected exactly one adjacent exon (number {adjacent_exon_number}), '
            f'found {len(adjacent_exon)}.'
        )
    exon_dict = adjacent_exon.to_dict(as_series=False)
    if direction == -1:
        new_ntpos = exon_dict['End'][0] - (row["Start"] - ntpos)
    else:
        new_ntpos = exon_dict['Start'][0] + (ntpos - row["End"])
    return str(ref_genome[exon_dict['Chromosome'][0]][new_ntpos])


def _analyze_single_cds_edit(
    edit: int,
    position: int,
    row: dict[str, Any],
    gene_symbol: str,
    transcript_symbol: str,
    first_transcript_exon: int,
    last_transcript_exon: int,
    exon_number: int,
    chrom: str,
    alt_edited: str,
    be_fwd_alt: str,
    non_stop_aas: list[str],
    codon_sun_one_letter: dict[str, str],
    cdss: pl.DataFrame,
    ref_genome: pyfaidx.Fasta,
) -> dict[str, Any]:
    """Compute codon, amino-acid, and consequence for a single edit position.

    Returns a dict with keys: codon, codon_edited, aa, aa_edited, consequence,
    aa_position, includes_splice_site, is_synonymous.
    """
    is_splice_site = False
    is_synonymous = False

    if row["Start"] <= edit < row["End"]:
        # Edit is inside the CDS
        if str(row['Strand']) == '+':
            index_edit_cds = edit - row['Start']
        else:
            index_edit_cds = row['End'] - 1 - edit

        offset = shared.get_offset(str(row['Strand']), int(row['Frame']), index_edit_cds)

        codon_in_cds = (row["Start"] <= (edit - offset) < row["End"]) and \
                       (row["Start"] <= (edit + 3 - offset) < row["End"])

        if codon_in_cds:
            codon = str(ref_genome[row["Chromosome"]][edit - offset:edit + 3 - offset])
            codon_edited = shared.replace_str_index(codon, offset, alt_edited)
        else:
            # Codon spans an exon boundary – fetch bases from adjacent exons
            try:
                codon_list = []
                for nt in range(edit - offset, edit + 3 - offset):
                    if nt < row["Start"]:
                        base = _lookup_adjacent_exon_base(
                            nt, row, exon_number, gene_symbol, transcript_symbol,
                            first_transcript_exon, last_transcript_exon, chrom,
                            cdss, ref_genome, direction=-1,
                        )
                    elif row["Start"] <= nt < row["End"]:
                        base = str(ref_genome[row["Chromosome"]][nt])
                    else:
                        base = _lookup_adjacent_exon_base(
                            nt, row, exon_number, gene_symbol, transcript_symbol,
                            first_transcript_exon, last_transcript_exon, chrom,
                            cdss, ref_genome, direction=+1,
                        )
                    codon_list.append(base)
                codon = ''.join(codon_list)
                codon_edited = shared.replace_str_index(codon, offset, be_fwd_alt)
            except (ValueError, IndexError, KeyError):
                codon = "incomplete_codon"
                codon_edited = "incomplete_codon"

        if str(row['Strand']) == '-':
            codon = shared.revcom(codon)
            codon_edited = shared.revcom(codon_edited)

        aa = codon_sun_one_letter.get(codon, codon)
        aa_edited = codon_sun_one_letter.get(codon_edited, codon_edited)

        # Mark start codon
        is_start = (
            aa == codon_sun_one_letter.get("ATG")
            and exon_number == first_transcript_exon
        )
        if is_start:
            in_start_codon = (
                (row['Strand'] == '+' and row["Start"] <= edit < row["Start"] + 3) or
                (row['Strand'] == '-' and row["End"] - 3 <= edit < row["End"])
            )
            if in_start_codon:
                codon = "Start" + codon
                aa = "Start" + aa

        consequence = _determine_consequence(aa, aa_edited, non_stop_aas)
        is_synonymous = aa == aa_edited

        if str(row['Strand']) == "+":
            nt_position = row['transcript_length_before'] + ((edit + 1) - row['Start'])
        else:
            nt_position = row['transcript_length_before'] + (row['End'] - edit)
        aa_position = str(-(-nt_position // 3))  # ceiling division

    elif (row["Start"] - 2) <= edit < (row["End"] + 2):
        # Edit is in a splice site
        is_splice_site = True
        is_synonymous = False
        consequence = 'SPLICE_SITE'

        is_before_exon = (row["Start"] - 2) <= edit < row["Start"]
        is_after_exon = row["End"] <= edit < (row["End"] + 2)

        if is_before_exon:
            if row["Strand"] == '+':
                codon = aa = "5prime_splice_site"
                if exon_number == first_transcript_exon:
                    codon = aa = "5prime_UTR"
                    is_splice_site = False
                    consequence = 'UTR'
            else:
                codon = aa = "3prime_splice_site"
                if exon_number == last_transcript_exon:
                    codon = aa = "3prime_UTR"
                    is_splice_site = False
                    consequence = 'UTR'
        else:  # is_after_exon
            if row["Strand"] == '-':
                codon = aa = "5prime_splice_site"
                if exon_number == first_transcript_exon:
                    codon = aa = "5prime_UTR"
                    is_splice_site = False
                    consequence = 'UTR'
            else:
                codon = aa = "3prime_splice_site"
                if exon_number == last_transcript_exon:
                    codon = aa = "3prime_UTR"
                    is_splice_site = False
                    consequence = 'UTR'

        codon_edited = codon
        aa_edited = aa
        aa_position = aa  # splice sites use the label as position too

    else:
        codon = codon_edited = aa = aa_edited = "not_in_CDS"
        consequence = "NOT_CDS"
        aa_position = aa

    # Lower-case fields for bystander edits (not the target variant position)
    if edit != position:
        codon = codon.lower()
        codon_edited = codon_edited.lower()
        aa = aa.lower()
        aa_edited = aa_edited.lower()
        consequence = consequence.lower()

    return {
        'codon': codon,
        'codon_edited': codon_edited,
        'aa': aa,
        'aa_edited': aa_edited,
        'consequence': consequence,
        'aa_position': aa_position,
        'includes_splice_site': is_splice_site,
        'synonymous': is_synonymous,
    }


def _annotate_cds_for_guide(
    total_poss: list[int],
    position: int,
    cdss: pl.DataFrame,
    chrom: str,
    ref_genome: pyfaidx.Fasta,
    be_name: str,
    bes: dict[str, Any],
    non_stop_aas: list[str],
    codon_sun_one_letter: dict[str, str],
    alt_edited: str,
) -> _GuideAnnotation:
    """Build the full CDS annotation for all edit positions of one guide."""
    # Filter CDS rows overlapping the variant position (with splice site margin)
    cdss_filtered = cdss.filter(
        (pl.col('Chromosome') == chrom) &
        (pl.col('Start') - 2 <= position) &
        (position < pl.col('End') + 2)
    )

    if cdss_filtered.is_empty():
        logger.debug("No CDS annotation found for %s:%d", chrom, position + 1)
        no_cds = ['no_CDS_found']
        return _GuideAnnotation(
            gene_symbols=no_cds,
            transcript_symbols=no_cds,
            exon_numbers=no_cds,
            first_transcript_exons=no_cds,
            last_transcript_exons=no_cds,
            codonss=[['no_CDS_found']],
            codonss_edited=[['no_CDS_found']],
            aass=[['no_CDS_found']],
            aass_edited=[['no_CDS_found']],
            aa_positionss=[['no_CDS_found']],
            splice_sites_included=['no_CDS_found'],
            synonymouss=['no_CDS_found'],
            consequences=[['no_CDS_found']],
        )

    gene_symbols: list[str] = []
    transcript_symbols: list[str] = []
    exon_numbers: list[str] = []
    first_transcript_exons: list[str] = []
    last_transcript_exons: list[str] = []
    codonss: list[list[str]] = []
    codonss_edited: list[list[str]] = []
    aass: list[list[str]] = []
    aass_edited: list[list[str]] = []
    aa_positionss: list[list[str]] = []
    splice_sites_included: list[str] = []
    synonymouss: list[str] = []
    consequences: list[list[str]] = []

    for row in cdss_filtered.iter_rows(named=True):
        gene_symbol = row['gene_name']
        transcript_symbol = row['transcript_name']
        exon_number = row['exon_number']
        first_transcript_exon = row['first_transcript_exon']
        last_transcript_exon = row['last_transcript_exon']

        per_exon_codons: list[str] = []
        per_exon_codons_edited: list[str] = []
        per_exon_aas: list[str] = []
        per_exon_aas_edited: list[str] = []
        per_exon_consequences: list[str] = []
        per_exon_aa_positions: list[str] = []
        includes_splice_site = False
        is_synonymous = False

        for edit in total_poss:
            result = _analyze_single_cds_edit(
                edit=edit,
                position=position,
                row=row,
                gene_symbol=gene_symbol,
                transcript_symbol=transcript_symbol,
                first_transcript_exon=first_transcript_exon,
                last_transcript_exon=last_transcript_exon,
                exon_number=exon_number,
                chrom=chrom,
                alt_edited=alt_edited,
                be_fwd_alt=bes[be_name]['fwd']['ALT'],
                non_stop_aas=non_stop_aas,
                codon_sun_one_letter=codon_sun_one_letter,
                cdss=cdss,
                ref_genome=ref_genome,
            )
            per_exon_codons.append(result['codon'])
            per_exon_codons_edited.append(result['codon_edited'])
            per_exon_aas.append(result['aa'])
            per_exon_aas_edited.append(result['aa_edited'])
            per_exon_consequences.append(result['consequence'])
            per_exon_aa_positions.append(result['aa_position'])
            includes_splice_site = includes_splice_site or result['includes_splice_site']

            # synonymous is only meaningful when there is a single edit
            if len(total_poss) == 1:
                is_synonymous = result['synonymous']

        gene_symbols.append(gene_symbol)
        transcript_symbols.append(transcript_symbol)
        exon_numbers.append(str(int(exon_number)))
        first_transcript_exons.append(str(int(first_transcript_exon)))
        last_transcript_exons.append(str(int(last_transcript_exon)))
        codonss.append(per_exon_codons)
        codonss_edited.append(per_exon_codons_edited)
        aass.append(per_exon_aas)
        aass_edited.append(per_exon_aas_edited)
        consequences.append(per_exon_consequences)
        aa_positionss.append(per_exon_aa_positions)
        splice_sites_included.append(str(includes_splice_site))
        synonymouss.append(str(is_synonymous))

    # Group overlapping transcript annotations
    annotations_df = pl.DataFrame({
        'gene_symbolss': gene_symbols,
        'transcript_symbolss': transcript_symbols,
        'exon_numberss': exon_numbers,
        'first_transcript_exonss': first_transcript_exons,
        'last_transcript_exonss': last_transcript_exons,
        'codonsss': codonss,
        'codonsss_edited': codonss_edited,
        'aasss': aass,
        'aasss_edited': aass_edited,
        'consequencess': consequences,
        'aa_positionsss': aa_positionss,
        'splice_sitess_included': splice_sites_included,
        'synonymousss': synonymouss,
    })

    group_keys = [
        col for col in annotations_df.columns
        if col not in [
            'aa_positionsss', 'exon_numberss', 'transcript_symbolss',
            'first_transcript_exonss', 'last_transcript_exonss',
        ]
    ]
    annotations_df = annotations_df.group_by(group_keys, maintain_order=True).agg(pl.all())

    return _GuideAnnotation(
        gene_symbols=annotations_df['gene_symbolss'].to_list(),
        transcript_symbols=annotations_df['transcript_symbolss'].to_list(),
        exon_numbers=annotations_df['exon_numberss'].to_list(),
        first_transcript_exons=annotations_df['first_transcript_exonss'].to_list(),
        last_transcript_exons=annotations_df['last_transcript_exonss'].to_list(),
        codonss=annotations_df['codonsss'].to_list(),
        codonss_edited=annotations_df['codonsss_edited'].to_list(),
        aass=annotations_df['aasss'].to_list(),
        aass_edited=annotations_df['aasss_edited'].to_list(),
        aa_positionss=annotations_df['aa_positionsss'].to_list(),
        splice_sites_included=annotations_df['splice_sitess_included'].to_list(),
        synonymouss=annotations_df['synonymousss'].to_list(),
        consequences=annotations_df['consequencess'].to_list(),
    )


# ---------------------------------------------------------------------------
# Core per-variant processing
# ---------------------------------------------------------------------------

def _compute_strand_string(rev_com: bool, fiveprimepam: bool) -> str:
    """Return the strand symbol given *rev_com* and *fiveprimepam* flags."""
    if fiveprimepam:
        return '+' if rev_com else '-'
    return '-' if rev_com else '+'


def _process_variant_with_editor(
    variant: str,
    be_name: str,
    bes: dict[str, Any],
    edits: dict[str, str],
    pam_config: _PamConfig,
    guidelength: int,
    ew_start: int,
    ew_end: int,
    ew_start_plus: int,
    ew_end_plus: int,
    allpossible: bool,
    fiveprimepam: bool,
    ignorestring: str | None,
    ref_genome: pyfaidx.Fasta,
    cdss: pl.DataFrame,
    distance_median_dict: dict[int, float],
    non_stop_aas: list[str],
    codon_sun_one_letter: dict[str, str],
) -> _VariantEditorResult:
    """Process a single (variant, base-editor) pair.

    This function is intentionally self-contained so that, in the future,
    it can be dispatched to a worker thread or process with no changes.
    """
    logger.debug("Processing variant %s with editor %s", variant, be_name)
    all_genes_transcripts = set(cdss.get_column('gene_name')) | set(cdss.get_column('transcript_name'))
    chrom, position_str, ref, alt, variant = _parse_variant_coords(variant, ignorestring, all_genes_transcripts)

    error_code = _variant_error_code(variant)

    # Convert VCF 1-based position to 0-based unless this variant has an error
    if error_code is None:
        if not position_str.isnumeric():
            variant = variant + ':genomic_position_not_numeric'
            error_code = 'genomic_position_not_numeric'
            position = 0
            logger.debug("Variant %s has a non-numeric genomic position", variant)
        else:
            position = int(position_str) - 1
    else:
        position = 0

    # Determine reference base at the target position
    if error_code is not None:
        target_base_ref = error_code
    else:
        target_base_ref, variant = _get_target_base(ref_genome, chrom, position, variant)
        if target_base_ref == 'genomic_coordinates_not_found':
            error_code = 'genomic_coordinates_not_found'

    logger.debug(
        "Variant %s resolved to chrom=%s position=%s ref=%s alt=%s target_base=%s",
        variant,
        chrom,
        position_str,
        ref or 'NA',
        alt,
        target_base_ref,
    )

    # Reference-match check
    if ref:
        target_seq_ref_match = "ref_match" if ref == target_base_ref else "REF_MISMATCH"
    else:
        target_seq_ref_match = "no_ref_input"

    # Determine the edited base for output (used in downstream annotation)
    if target_base_ref in _ERROR_CODES:
        alt_edited = 'be_not_usable'
    elif allpossible:
        alt_edited = edits.get(target_base_ref, 'be_not_usable')
    else:
        alt_edited = alt

    error_code = _variant_error_code(variant)  # re-check after possible update

    # Compute the variant_real string
    all_genes_transcripts = set(cdss.get_column('gene_name')) | set(cdss.get_column('transcript_name'))
    variant_real = _extract_variant_real(
        variant, error_code, target_base_ref, edits, allpossible, all_genes_transcripts
    )

    # Non-editable early exit
    if error_code is not None or target_base_ref in _ERROR_CODES:
        logger.debug(
            "Skipping variant %s with editor %s due to error condition %s",
            variant,
            be_name,
            error_code or target_base_ref,
        )
        return _VariantEditorResult(
            variant=variant,
            variant_real=variant_real,
            editable=False,
            be_string=be_name,
            original_alt=False,
            target_seq_ref="",
            target_seq_ref_match=target_seq_ref_match,
            target_base_ref=target_base_ref,
            target_base="",
            strand=_compute_strand_string(False, fiveprimepam),
            guides=[],
        )

    # Check editability
    editability = _determine_editability(
        target_base_ref=target_base_ref,
        alt=alt,
        be_fwd=bes[be_name]['fwd'],
        be_rev=bes[be_name]['rev'],
        be_name=be_name,
        allpossible=allpossible,
        pam_config=pam_config,
        position=position,
        ref_genome=ref_genome,
        chrom=chrom,
    )

    if editability is None:
        logger.debug("Variant %s is not editable by %s", variant_real, be_name)
        return _VariantEditorResult(
            variant=variant,
            variant_real=variant_real,
            editable=False,
            be_string=be_name,
            original_alt=False,
            target_seq_ref="",
            target_seq_ref_match=target_seq_ref_match,
            target_base_ref=target_base_ref,
            target_base="",
            strand=_compute_strand_string(False, fiveprimepam),
            guides=[],
        )

    # Find all possible guides
    target_seq = editability.target_seq
    variant_position = pam_config.bases_before_variant
    target_seq_ref_start = editability.target_seq_ref_start
    guide_records: list[_GuideRecord] = []

    for i in range(
        len(target_seq) - pam_config.start_of_pam_search,
        len(target_seq) - pam_config.end_of_pam_search,
    ):
        pam_candidate = target_seq[i: i + pam_config.length_of_pam_search]
        if not any(pam_candidate == s for s in pam_config.pamlist_real_string):
            variant_position -= 1
            if editability.rev_com:
                target_seq_ref_start -= 1
            else:
                target_seq_ref_start += 1
            continue

        guide_with_pam = target_seq[
            i - pam_config.start_before_hit: i + pam_config.end_after_hit_with_pam
        ]
        guide = target_seq[
            i - pam_config.start_before_hit: i - pam_config.end_after_hit_guide_only
        ]
        pam = target_seq[
            i - pam_config.count_ignored_n_in_pam: i + pam_config.count_of_other_in_pam
        ]

        (
            edit_window, num_edits, specific,
            edit_window_plus, num_edits_plus, specific_plus,
            safety_region, num_edits_safety, additional_in_safety,
            edit_string, edit_pos_string,
            specificity, distance_to_center_variant, distance_to_center,
        ) = shared.analyze_guide(
            guide, ew_start, ew_end, ew_start_plus, ew_end_plus,
            editability.target_base, variant_position,
            distance_median_dict, fiveprimepam,
        )

        # Build list of genomic positions affected by this guide
        edit_string_genomic = edit_string[::-1] if editability.rev_com else edit_string
        edit_list = []
        variant_pos_rel = None
        for idx, char in enumerate(edit_string_genomic):
            if char in ('*', 'V'):
                edit_list.append(idx)
                if char == 'V':
                    variant_pos_rel = idx

        total_poss = [position - (variant_pos_rel - m) for m in edit_list]

        var_with_by = ';'.join(
            '_'.join([chrom, str(p + 1), target_base_ref, edits[target_base_ref]])
            for p in total_poss
        )

        # Flip sequences for 5'-PAM output
        if fiveprimepam:
            guide = shared.revcom(guide)
            guide_with_pam = shared.revcom(guide_with_pam)
            pam = shared.revcom(pam)
            edit_window = shared.revcom(edit_window)
            edit_window_plus = shared.revcom(edit_window_plus)
            safety_region = shared.revcom(safety_region)
            edit_string = edit_string[::-1]
            edit_pos_string = edit_pos_string[::-1]
            distance_to_center_variant = distance_to_center_variant[::-1]
            distance_to_center = distance_to_center[::-1]

        # Genomic start/end in 1-based coordinates (for SAM output)
        if editability.rev_com:
            guide_start = str(target_seq_ref_start - guidelength + 1)
            guide_end = str(target_seq_ref_start + 1)
        else:
            guide_start = str(target_seq_ref_start + 1)
            guide_end = str(target_seq_ref_start + guidelength + 1)

        annotation = _annotate_cds_for_guide(
            total_poss=total_poss,
            position=position,
            cdss=cdss,
            chrom=chrom,
            ref_genome=ref_genome,
            be_name=be_name,
            bes=bes,
            non_stop_aas=non_stop_aas,
            codon_sun_one_letter=codon_sun_one_letter,
            alt_edited=alt_edited,
        )

        guide_records.append(_GuideRecord(
            guide_with_pam=guide_with_pam,
            guide=guide,
            pam=pam,
            guide_start=guide_start,
            guide_end=guide_end,
            guide_chrom=str(chrom),
            var_with_by=var_with_by,
            edit_window=edit_window,
            num_edits=str(num_edits),
            specific=str(specific),
            edit_window_plus=edit_window_plus,
            num_edits_plus=str(num_edits_plus),
            specific_plus=str(specific_plus),
            safety_region=safety_region,
            num_edits_safety=str(num_edits_safety),
            additional_in_safety=str(additional_in_safety),
            edit_string=edit_string,
            edit_pos_string=edit_pos_string,
            specificity=str(specificity),
            distance_to_center_variant=distance_to_center_variant,
            distance_to_center=distance_to_center,
            annotation=annotation,
        ))

        variant_position -= 1
        if editability.rev_com:
            target_seq_ref_start -= 1
        else:
            target_seq_ref_start += 1

    if guide_records:
        logger.debug(
            "Variant %s with editor %s produced %d guide(s)",
            variant_real,
            be_name,
            len(guide_records),
        )
    else:
        logger.debug(
            "Variant %s is editable by %s but no compatible guides were found",
            variant_real,
            be_name,
        )

    return _VariantEditorResult(
        variant=variant,
        variant_real=variant_real,
        editable=True,
        be_string=editability.be_string,
        original_alt=editability.original_alt,
        target_seq_ref=editability.target_seq_ref,
        target_seq_ref_match=target_seq_ref_match,
        target_base_ref=target_base_ref,
        target_base=editability.target_base,
        strand=_compute_strand_string(editability.rev_com, fiveprimepam),
        guides=guide_records,
    )


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

_NO_GUIDES = "no_guides_found"
_NOT_USABLE = "be_not_usable"


def _build_sgrna_dataframe(results: list[_VariantEditorResult]) -> pl.DataFrame:
    """Convert a list of :class:`_VariantEditorResult` objects into a DataFrame.

    The nested structure mirrors what the original code built from parallel
    lists and is required by the downstream post-processing steps.
    """
    rows: dict[str, list[Any]] = {
        "variant": [], "base_change": [], "symbol": [], "guide": [],
        "guide_chrom": [], "guide_start": [], "guide_end": [],
        "guide_with_pam": [], "edit_window": [], "num_edits": [],
        "variant_and_bystanders": [], "specific": [], "edit_window_plus": [],
        "num_edits_plus": [], "specific_plus": [], "safety_region": [],
        "num_edits_safety": [], "additional_in_safety": [], "ne_plus": [],
        "synonymous_specific": [], "consequence": [], "strand": [],
        "codon_ref": [], "aa_ref": [], "aa_pos": [], "codon_edit": [],
        "aa_edit": [], "splice_site_included": [],
        "originally_intended_ALT": [], "ref_match": [],
        "off_target_bases": [], "edited_positions": [], "specificity": [],
        "distance_to_center_variant": [], "distance_to_center": [],
        "transcript": [], "exon_number": [], "first_transcript_exon": [],
        "last_transcript_exon": [],
    }

    for res in results:
        if not res.editable or not res.guides:
            # Non-editable or editable-but-no-guides placeholder
            marker = _NOT_USABLE if not res.editable else _NO_GUIDES
            rows["variant"].append(res.variant)
            rows["base_change"].append(res.be_string)
            rows["strand"].append(marker if not res.editable else res.strand)
            rows["originally_intended_ALT"].append(marker)
            rows["ref_match"].append(res.target_seq_ref_match)
            rows["ne_plus"].append("NA_for_variants")
            for key in ["guide", "guide_chrom", "guide_start", "guide_end",
                        "guide_with_pam", "edit_window", "num_edits",
                        "variant_and_bystanders", "specific", "edit_window_plus",
                        "num_edits_plus", "specific_plus", "safety_region",
                        "num_edits_safety", "additional_in_safety",
                        "off_target_bases", "edited_positions", "specificity",
                        "distance_to_center_variant", "distance_to_center"]:
                rows[key].append([marker])
            if not res.editable:
                rows["symbol"].append([["be_not_usable"]])
                rows["transcript"].append([[["be_not_usable"]]])
                rows["exon_number"].append([[["be_not_usable"]]])
                rows["first_transcript_exon"].append([[["be_not_usable"]]])
                rows["last_transcript_exon"].append([[["be_not_usable"]]])
                rows["codon_ref"].append([[["be_not_usable"]]])
                rows["codon_edit"].append([[["be_not_usable"]]])
                rows["aa_ref"].append([[["be_not_usable"]]])
                rows["aa_edit"].append([[["be_not_usable"]]])
                rows["aa_pos"].append([[[["be_not_usable"]]]])
                rows["splice_site_included"].append([["be_not_usable"]])
                rows["synonymous_specific"].append([["be_not_usable"]])
                rows["consequence"].append([[["be_not_usable"]]])
            else:
                rows["symbol"].append([["no_guides_found"]])
                rows["transcript"].append([[["no_guides_found"]]])
                rows["exon_number"].append([[["no_guides_found"]]])
                rows["first_transcript_exon"].append([[["no_guides_found"]]])
                rows["last_transcript_exon"].append([[["no_guides_found"]]])
                rows["codon_ref"].append([[["no_guides_found"]]])
                rows["codon_edit"].append([[["no_guides_found"]]])
                rows["aa_ref"].append([[["no_guides_found"]]])
                rows["aa_edit"].append([[["no_guides_found"]]])
                rows["aa_pos"].append([[[["no_guides_found"]]]])
                rows["splice_site_included"].append([["no_guides_found"]])
                rows["synonymous_specific"].append([["no_guides_found"]])
                rows["consequence"].append([[["no_guides_found"]]])
            continue

        guides = res.guides
        rows["variant"].append(res.variant)
        rows["base_change"].append(res.be_string)
        rows["strand"].append(res.strand)
        rows["originally_intended_ALT"].append(res.original_alt)
        rows["ref_match"].append(res.target_seq_ref_match)
        rows["ne_plus"].append("NA_for_variants")

        rows["guide"].append([g.guide for g in guides])
        rows["guide_chrom"].append([g.guide_chrom for g in guides])
        rows["guide_start"].append([g.guide_start for g in guides])
        rows["guide_end"].append([g.guide_end for g in guides])
        rows["guide_with_pam"].append([g.guide_with_pam for g in guides])
        rows["edit_window"].append([g.edit_window for g in guides])
        rows["num_edits"].append([g.num_edits for g in guides])
        rows["variant_and_bystanders"].append([g.var_with_by for g in guides])
        rows["specific"].append([g.specific for g in guides])
        rows["edit_window_plus"].append([g.edit_window_plus for g in guides])
        rows["num_edits_plus"].append([g.num_edits_plus for g in guides])
        rows["specific_plus"].append([g.specific_plus for g in guides])
        rows["safety_region"].append([g.safety_region for g in guides])
        rows["num_edits_safety"].append([g.num_edits_safety for g in guides])
        rows["additional_in_safety"].append([g.additional_in_safety for g in guides])
        rows["off_target_bases"].append([g.edit_string for g in guides])
        rows["edited_positions"].append([g.edit_pos_string for g in guides])
        rows["specificity"].append([g.specificity for g in guides])
        rows["distance_to_center_variant"].append([g.distance_to_center_variant for g in guides])
        rows["distance_to_center"].append([g.distance_to_center for g in guides])

        rows["symbol"].append([g.annotation.gene_symbols for g in guides])
        rows["transcript"].append([g.annotation.transcript_symbols for g in guides])
        rows["exon_number"].append([g.annotation.exon_numbers for g in guides])
        rows["first_transcript_exon"].append([g.annotation.first_transcript_exons for g in guides])
        rows["last_transcript_exon"].append([g.annotation.last_transcript_exons for g in guides])
        rows["codon_ref"].append([g.annotation.codonss for g in guides])
        rows["codon_edit"].append([g.annotation.codonss_edited for g in guides])
        rows["aa_ref"].append([g.annotation.aass for g in guides])
        rows["aa_edit"].append([g.annotation.aass_edited for g in guides])
        rows["aa_pos"].append([g.annotation.aa_positionss for g in guides])
        rows["splice_site_included"].append([g.annotation.splice_sites_included for g in guides])
        rows["synonymous_specific"].append([g.annotation.synonymouss for g in guides])
        rows["consequence"].append([g.annotation.consequences for g in guides])

    sgrna_df = pl.DataFrame(rows, strict=False)
    logger.debug("Built sgRNA dataframe with %d row(s)", sgrna_df.height)
    return sgrna_df


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _apply_blast_annotations(
    sgrnas: pl.DataFrame,
    guidelength: int,
    refgenome: str,
    no_contigs: bool,
) -> pl.DataFrame:
    """Add BLAST off-target counts to *sgrnas*."""
    logger.info("Running BLAST annotation for %d design row(s)", sgrnas.height)
    blast_guides.check_blastdb(refgenome, False)
    sgrnas = sgrnas.with_row_index('index')
    guides = sgrnas.select('index', 'guide').explode('guide')
    blast_results = blast_guides.guide_blast(guides, guidelength, refgenome, 'variants', no_contigs)
    if not blast_results.is_empty():
        sgrnas = sgrnas.join(blast_results, left_on='index', right_on='indexvar', how='left')
        sgrnas = sgrnas.with_columns(
            blastcount=pl.when(
                (pl.col('guide') == ["no_guides_found"]) | (pl.col('guide') == ["be_not_usable"])
            ).then(pl.col('guide')).otherwise(pl.col('blastcount'))
        )
    else:
        sgrnas = sgrnas.with_columns(blastcount=pl.col('guide'))
    logger.info("Finished BLAST annotation")
    return sgrnas.drop('index')


def _apply_vep_annotations(
    sgrnas: pl.DataFrame,
    all_vars_with_bys: list[Any],
    vep_species: str,
    vep_assembly: str,
    vep_dir_cache: str,
    vep_cache_version: str,
    vep_flags: str,
) -> pl.DataFrame:
    """Fetch and merge VEP annotations into *sgrnas*.

    Note: the ``vep_dir_plugins`` parameter exists in the public API for
    forward-compatibility but is not yet consumed by VEP (commented out
    in the original implementation).
    """
    non_vep_values = list(_ERROR_CODES) + ['no_guides_found', 'be_not_usable']

    variants_vep = (
        pl.DataFrame({'variant': all_vars_with_bys})
        .with_row_index('for_sorting_third')
        .explode("variant")
        .with_row_index('for_sorting_second')
        .with_columns(pl.col("variant").str.split(";").alias("variant"))
        .explode("variant")
        .with_row_index('for_sorting_first')
    )

    variants_non_vep = variants_vep.filter(pl.col('variant').is_in(non_vep_values))
    variants_vep = variants_vep.filter(~pl.col('variant').is_in(non_vep_values))

    variants_vep_sorted = shared.sort_variantsdf(variants_vep)
    unique_variants = variants_vep_sorted['variant'].unique(maintain_order=True).to_list()

    logger.info("Running VEP annotation for %d unique variant(s)", len(unique_variants))

    vep_annotations = get_vep.get_vep_annotation(
        unique_variants,
        species=vep_species,
        assembly=vep_assembly,
        dir_cache=vep_dir_cache,
        cache_version=vep_cache_version,
        flags=vep_flags,
    )
    vep_annotations = vep_annotations.rename(lambda col: "VEP_" + col)
    vep_annotations = vep_annotations.group_by('VEP_#Uploaded_variation').agg(
        pl.all().str.join(",")
    )
    vep_annotations = vep_annotations.with_columns(
        pl.col('VEP_#Uploaded_variation')
        .str.replace('./', '', literal=True)
        .str.replace('/', '_', literal=True)
    )

    variants_vep_sorted = variants_vep_sorted.join(
        vep_annotations, left_on='variant', right_on='VEP_#Uploaded_variation', how='left'
    )
    variants_vep_resorted = shared.resort_variantsdf(variants_vep_sorted)

    for col in [c for c in vep_annotations.columns if c in variants_vep_resorted.columns]:
        variants_non_vep = variants_non_vep.with_columns(
            pl.lit("not_suitable_for_VEP").alias(col)
        )

    variants_vep_resorted = (
        pl.concat([variants_vep_resorted, variants_non_vep])
        .sort(['for_sorting_first', 'for_sorting_second', 'for_sorting_third'])
        .drop('for_sorting_first')
        .group_by(['for_sorting_second', 'for_sorting_third'], maintain_order=True)
        .agg(pl.all())
        .with_columns(pl.exclude(['for_sorting_second', 'for_sorting_third']).list.join(";"))
        .sort(['for_sorting_second', 'for_sorting_third'])
        .drop('for_sorting_second')
        .group_by('for_sorting_third', maintain_order=True)
        .agg(pl.all())
        .drop('for_sorting_third')
    )

    variants_vep_resorted = variants_vep_resorted.select(
        pl.exclude(["original_index", "variant"])
    )
    logger.info("Finished VEP annotation")
    return sgrnas.with_columns(variants_vep_resorted)


def _apply_filters(
    sgrnas: pl.DataFrame,
    filter_synonymous: bool,
    filter_splice_site: bool,
    filter_specific: bool,
    filter_missense: bool,
    filter_nonsense: bool,
    filter_stoplost: bool,
    filter_startlost: bool,
    symbols_to_contract: list[str],
    columns_to_modify_last: list[str],
) -> pl.DataFrame:
    """Filter *sgrnas* according to the active filter flags."""
    initial_rows = sgrnas.height
    active_filters = [
        name for name, enabled in [
            ('synonymous', filter_synonymous),
            ('splice_site', filter_splice_site),
            ('specific', filter_specific),
            ('missense', filter_missense),
            ('nonsense', filter_nonsense),
            ('stoplost', filter_stoplost),
            ('startlost', filter_startlost),
        ]
        if enabled
    ]
    logger.info("Applying filters: %s", ', '.join(active_filters))

    sgrnas_columns = sgrnas.columns
    sgrnas = sgrnas.explode(symbols_to_contract + columns_to_modify_last)

    sgrnas_not_to_filter = sgrnas.filter(
        (pl.col("guide").cast(str) == "no_guides_found") |
        (pl.col("guide").cast(str) == "be_not_usable")
    )
    sgrnas_to_filter = sgrnas.filter(
        (pl.col("guide").cast(str) != "no_guides_found") &
        (pl.col("guide").cast(str) != "be_not_usable")
    )

    if filter_synonymous:
        sgrnas_to_filter = sgrnas_to_filter.filter(
            ~(
                pl.col('synonymous_specific').str.contains("False") |
                pl.col('synonymous_specific').str.contains("false")
            )
        )
    if filter_splice_site:
        sgrnas_to_filter = sgrnas_to_filter.filter(
            pl.col('consequence').str.contains("splice_site") |
            pl.col('consequence').str.contains("SPLICE_SITE")
        )
    if filter_specific:
        sgrnas_to_filter = sgrnas_to_filter.filter(
            (pl.col('specific') == "True") | (pl.col('specific') == "true")
        )
    if filter_missense:
        sgrnas_to_filter = sgrnas_to_filter.filter(
            pl.col('consequence').str.contains("missense") |
            pl.col('consequence').str.contains("MISSENSE")
        )
    if filter_nonsense:
        sgrnas_to_filter = sgrnas_to_filter.filter(
            pl.col('consequence').str.contains("stopgain") |
            pl.col('consequence').str.contains("STOPGAIN")
        )
    if filter_stoplost:
        sgrnas_to_filter = sgrnas_to_filter.filter(
            pl.col('consequence').str.contains("stoplost") |
            pl.col('consequence').str.contains("STOPLOST")
        )
    if filter_startlost:
        sgrnas_to_filter = sgrnas_to_filter.filter(
            pl.col('consequence').str.contains("startlost") |
            pl.col('consequence').str.contains("STARTLOST")
        )

    def _variant_be_key(df: pl.DataFrame) -> list[str]:
        return df.with_columns(
            pl.concat_str([pl.col("variant"), pl.col("base_change")], separator="_")
            .alias("variant_base_editor")
        )['variant_base_editor'].to_list()

    keep = set(_variant_be_key(sgrnas_not_to_filter)) | set(_variant_be_key(sgrnas_to_filter))

    sgrnas_filtered_out = sgrnas.with_columns(
        pl.concat_str([pl.col("variant"), pl.col("base_change")], separator="_")
        .alias("variant_base_editor")
    ).filter(
        ~pl.col('variant_base_editor').is_in(keep)
    ).drop('variant_base_editor')

    fill_cols = [
        c for c in sgrnas_filtered_out.columns
        if c not in ['variant', 'base_change', 'strand', 'ref_match']
    ]
    for col in fill_cols:
        sgrnas_filtered_out = sgrnas_filtered_out.with_columns(
            pl.lit("no_guides_found").alias(col)
        )

    sgrnas = pl.concat(
        [
            pl.concat([sgrnas_not_to_filter, sgrnas_filtered_out], how="vertical_relaxed").unique(),
            sgrnas_to_filter,
        ],
        how="vertical_relaxed",
    )
    sgrnas = sgrnas.group_by(
        [c for c in sgrnas.columns if c not in (symbols_to_contract + columns_to_modify_last)],
        maintain_order=True,
    ).agg(pl.all())
    logger.info(
        "Filtering completed: %d row(s) before filters, %d row(s) after filters",
        initial_rows,
        sgrnas.height,
    )
    return sgrnas[sgrnas_columns]


def _format_output(
    sgrnas: pl.DataFrame,
    allpossible: bool,
    all_variant_real: list[str],
    aspect: str,
    guidelength: int,
    ew_start_plus: int,
    ew_end_plus: int,
    symbols_to_contract: list[str],
    columns_to_modify_last: list[str],
    non_list_columns: list[str],
    any_filter: bool,
    filter_synonymous: bool,
    filter_splice_site: bool,
    filter_specific: bool,
    filter_missense: bool,
    filter_nonsense: bool,
    filter_stoplost: bool,
    filter_startlost: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply final transforms and produce SAM-ready data alongside the main table."""
    logger.debug("Formatting output in %s aspect", aspect)
    # Column groupings for nested-list flattening
    aa_pos_cols = ['aa_pos']
    splice_site_cols = ['splice_site_included', 'synonymous_specific']
    consequence_cols = ['consequence']
    transcript_cols = ['exon_number', 'transcript', 'first_transcript_exon', 'last_transcript_exon']
    variant_annotation_cols = ['codon_ref', 'aa_ref', 'codon_edit', 'aa_edit']

    sgrnas = sgrnas.with_columns(
        pl.col(aa_pos_cols).list.eval(
            pl.element()
            .list.eval(pl.element().list.eval(pl.element().list.join(";")).list.join("~"))
            .list.join("^")
        ),
        pl.col(symbols_to_contract).list.eval(
            pl.element().list.unique(maintain_order=True).list.join("?")
        ),
        pl.col(transcript_cols).list.eval(
            pl.element().list.eval(pl.element().list.join("~")).list.join("^")
        ),
        pl.col(variant_annotation_cols).list.eval(
            pl.element().list.eval(pl.element().list.join(";")).list.join("^")
        ),
        pl.col(splice_site_cols).list.eval(pl.element().list.join("^")),
        pl.col(consequence_cols).list.eval(
            pl.element().list.eval(pl.element().list.join(";")).list.join("^")
        ),
    )

    if allpossible:
        all_variant = sgrnas['variant'].to_list()
        sgrnas = sgrnas.with_columns(
            pl.Series('variant', [
                f'{all_variant[i]}({all_variant_real[i]})'
                if all_variant[i].split(':')[-1] != all_variant_real[i]
                else all_variant[i]
                for i in range(len(all_variant))
            ])
        )

    if any_filter:
        sgrnas = _apply_filters(
            sgrnas=sgrnas,
            filter_synonymous=filter_synonymous,
            filter_splice_site=filter_splice_site,
            filter_specific=filter_specific,
            filter_missense=filter_missense,
            filter_nonsense=filter_nonsense,
            filter_stoplost=filter_stoplost,
            filter_startlost=filter_startlost,
            symbols_to_contract=symbols_to_contract,
            columns_to_modify_last=columns_to_modify_last,
        )

    if aspect == 'exploded':
        sgrnas = sgrnas.explode(symbols_to_contract + columns_to_modify_last)
    elif aspect == 'collapsed':
        sgrnas = sgrnas.with_columns(
            pl.col(symbols_to_contract).list.unique(maintain_order=True).list.join("?"),
            pl.col(columns_to_modify_last).list.join("•"),
        )

    sgrnas = sgrnas.unique().sort(by=sgrnas.columns)

    # Build SAM-compatible data
    sam_df = sgrnas.filter(
        (pl.col("guide").cast(str) != "no_guides_found") &
        (pl.col("guide").cast(str) != "be_not_usable")
    ).select(
        ['variant', 'guide', 'guide_chrom', 'guide_start', 'strand']
    ).with_columns(
        pl.col("guide").str.split("•"),
        pl.col("guide_chrom").str.split("•"),
        pl.col("guide_start").str.split("•"),
    ).explode(['guide', 'guide_chrom', 'guide_start'])

    sam_df = sam_df.with_columns(
        pl.col('variant').alias('QNAME'),
        pl.when(pl.col("strand") == '+').then(0).otherwise(
            pl.when(pl.col("strand") == '-').then(16)
        ).alias('FLAG'),
        pl.col('guide_chrom').alias('RNAME'),
        pl.col('guide_start').cast(pl.Int64).alias('POS'),
        pl.lit(255).alias('MAPQ'),
        pl.lit(str(guidelength) + 'M').alias('CIGAR'),
        pl.lit('*').alias('RNEXT'),
        pl.lit(0).alias('PNEXT'),
        pl.lit(0).alias('TLEN'),
        pl.when(pl.col("strand") == '+').then(pl.col('guide')).otherwise(
            pl.when(pl.col("strand") == '-').then(
                pl.col('guide').map_elements(shared.revcom, return_dtype=pl.String)
            )
        ).alias('SEQ'),
        pl.lit('*').alias('QUAL'),
    ).select(
        ['QNAME', 'FLAG', 'RNAME', 'POS', 'MAPQ', 'CIGAR',
         'RNEXT', 'PNEXT', 'TLEN', 'SEQ', 'QUAL']
    ).sort(by=['RNAME', 'POS', 'QNAME', 'FLAG', 'MAPQ', 'CIGAR',
               'RNEXT', 'PNEXT', 'TLEN', 'SEQ', 'QUAL'])

    # Drop internal-only columns
    sgrnas = sgrnas.drop(['ne_plus', 'off_target_bases', 'specificity'])

    if not allpossible:
        sgrnas = sgrnas.drop(['originally_intended_ALT'])

    if ew_start_plus == 0 and ew_end_plus == 0:
        sgrnas = sgrnas.drop([
            'edit_window_plus', 'num_edits_plus', 'specific_plus',
            'safety_region', 'num_edits_safety', 'additional_in_safety',
        ])

    logger.debug(
        "Prepared formatted sgRNA table with %d row(s) and SAM table with %d row(s)",
        sgrnas.height,
        sam_df.height,
    )
    return sgrnas, sam_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def design_bes(
    annotation_file: str,
    refgenome: str,
    input_variant: str | None,
    input_file: str | None,
    pamsite: str,
    fiveprimepam: bool,
    edit_window_start: int,
    edit_window_end: int,
    guidelength: int,
    ignorestring: str | None,
    basechange: list[str],
    edit_window_start_plus: int,
    edit_window_end_plus: int,
    allpossible: bool,
    input_format: str | None,
    mane_select_only: bool,
    write_parquet: bool,
    vep: bool,
    aspect: str,
    vep_flags: str,
    vep_species: str,
    vep_assembly: str,
    vep_dir_cache: str,
    vep_dir_plugins: str,
    vep_cache_version: str,
    dbsnp_db: str,
    blast: bool,
    no_contigs: bool,
    filter_synonymous: bool,
    filter_splice_site: bool,
    filter_specific: bool,
    filter_missense: bool,
    filter_nonsense: bool,
    filter_stoplost: bool,
    filter_startlost: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Design base editing guides for one or more genomic variants.

    Parameters mirror the CLI arguments accepted by ``bescreen.py``.

    Returns
    -------
    (sgrnas, sam_df)
        *sgrnas* – one row per variant (collapsed) or guide (exploded).
        *sam_df* – SAM-format data for BAM file generation.
    """
    logger.info("Starting base editing guide design")

    if guidelength < 17:
        raise ValueError('Please set the guide length to at least 17 bp!')

    if not basechange:
        raise ValueError('Please select at least one base editor!')

    logger.debug(
        "Design parameters: pamsite=%s fiveprimepam=%s guide_length=%d window=%d-%d "
        "window_plus=%d-%d aspect=%s allpossible=%s mane_select_only=%s",
        pamsite,
        fiveprimepam,
        guidelength,
        edit_window_start,
        edit_window_end,
        edit_window_start_plus,
        edit_window_end_plus,
        aspect,
        allpossible,
        mane_select_only,
    )

    # Deep-copy shared state so this call is re-entrant (safe for threading)
    iupac_nt_code = copy.deepcopy(shared.iupac_nt_code)
    codon_sun_one_letter = copy.deepcopy(shared.codon_sun_one_letter)

    bes = copy.deepcopy(shared.bes)
    for be in bes:
        # Remove keys other than 'fwd' and 'rev' which would disturb iteration
        for key in [k for k in list(bes[be].keys()) if k not in ('fwd', 'rev')]:
            del bes[be][key]

    if fiveprimepam:
        # For 5'-PAM editors: swap fwd/rev edit directions, reverse PAM,
        # and invert the editing window coordinates
        bes = shared.fiveprimepam_bes(bes)
        pamsite = shared.revcom(pamsite)

        ew_start_new = guidelength - edit_window_end + 1
        edit_window_end = guidelength - edit_window_start + 1
        edit_window_start = ew_start_new

        ew_plus_start_new = edit_window_end_plus
        edit_window_end_plus = edit_window_start_plus
        edit_window_start_plus = ew_plus_start_new

        logger.debug("Adjusted editor configuration for 5' PAM mode")

    if 'all' not in basechange:
        bes = {key: bes[key] for key in basechange}

    logger.info("Using %d base editor(s)", len(bes))
    logger.debug("Selected base editors: %s", ', '.join(bes.keys()))

    logger.info("Loading reference genome from %s", refgenome)
    ref_genome_pyfaidx = pyfaidx.Fasta(refgenome)
    parquet_file = shared.check_parquet(annotation_file, write_parquet)
    cdss = pl.read_parquet(parquet_file)
    logger.info("Loaded annotation parquet %s with %d CDS row(s)", parquet_file, cdss.height)

    if mane_select_only:
        cdss = cdss.filter(pl.col('MANE_Select'))
        logger.info("Restricted annotation to %d MANE Select CDS row(s)", cdss.height)

    variant_list = _prepare_variant_list(
        input_variant, input_file, input_format, cdss, ref_genome_pyfaidx, dbsnp_db
    )
    logger.info(
        "Prepared %d variant input(s) across %d base editor(s)",
        len(variant_list),
        len(bes),
    )

    distance_median_dict = shared.qc_precalc(edit_window_start, edit_window_end)
    non_stop_aas = list({v for v in codon_sun_one_letter.values() if v != 'Stop'})
    any_filter = any([
        filter_synonymous, filter_splice_site, filter_specific,
        filter_missense, filter_nonsense, filter_stoplost, filter_startlost,
    ])

    pam_config = _setup_pam_config(
        pamsite, guidelength, edit_window_start, edit_window_end, iupac_nt_code
    )

    # Main design loop – structured so that the inner call to
    # _process_variant_with_editor can be parallelised in the future without
    # any changes to that function's signature.
    results: list[_VariantEditorResult] = []
    all_variant_real: list[str] = []

    for be_name in bes:
        edits: dict[str, str] = {base: 'be_not_usable' for base in 'ATCG'}
        for direction in bes[be_name]:
            edits[bes[be_name][direction]['REF']] = bes[be_name][direction]['ALT']

        for variant in variant_list:
            result = _process_variant_with_editor(
                variant=variant,
                be_name=be_name,
                bes=bes,
                edits=edits,
                pam_config=pam_config,
                guidelength=guidelength,
                ew_start=edit_window_start,
                ew_end=edit_window_end,
                ew_start_plus=edit_window_start_plus,
                ew_end_plus=edit_window_end_plus,
                allpossible=allpossible,
                fiveprimepam=fiveprimepam,
                ignorestring=ignorestring,
                ref_genome=ref_genome_pyfaidx,
                cdss=cdss,
                distance_median_dict=distance_median_dict,
                non_stop_aas=non_stop_aas,
                codon_sun_one_letter=codon_sun_one_letter,
            )
            results.append(result)
            all_variant_real.append(result.variant_real)

            logger.info("Processed %d variant/editor combination(s)", len(results))

    sgrnas = _build_sgrna_dataframe(results)

    # Collect vars_with_bys for optional VEP annotation
    all_vars_with_bys = sgrnas['variant_and_bystanders'].to_list()

    if blast:
        sgrnas = _apply_blast_annotations(sgrnas, guidelength, refgenome, no_contigs)

    if vep:
        sgrnas = _apply_vep_annotations(
            sgrnas=sgrnas,
            all_vars_with_bys=all_vars_with_bys,
            vep_species=vep_species,
            vep_assembly=vep_assembly,
            vep_dir_cache=vep_dir_cache,
            vep_cache_version=vep_cache_version,
            vep_flags=vep_flags,
        )

    # Column groupings used by format / filter helpers
    symbols_to_contract = ['symbol']
    non_list_columns = [
        'variant', 'symbol', 'base_change', 'ne_plus', 'strand',
        'originally_intended_ALT', 'ref_match',
    ]
    columns_to_modify_last = [c for c in sgrnas.columns if c not in non_list_columns]

    sgrnas, sam_df = _format_output(
        sgrnas=sgrnas,
        allpossible=allpossible,
        all_variant_real=all_variant_real,
        aspect=aspect,
        guidelength=guidelength,
        ew_start_plus=edit_window_start_plus,
        ew_end_plus=edit_window_end_plus,
        symbols_to_contract=symbols_to_contract,
        columns_to_modify_last=columns_to_modify_last,
        non_list_columns=non_list_columns,
        any_filter=any_filter,
        filter_synonymous=filter_synonymous,
        filter_splice_site=filter_splice_site,
        filter_specific=filter_specific,
        filter_missense=filter_missense,
        filter_nonsense=filter_nonsense,
        filter_stoplost=filter_stoplost,
        filter_startlost=filter_startlost,
    )

    logger.info(
        "Finished guide design with %d output row(s) and %d SAM row(s)",
        sgrnas.height,
        sam_df.height,
    )

    return sgrnas, sam_df


def output_sgrnas(sgrnas: pl.DataFrame, output_file: str) -> None:
    """Write *sgrnas* to TSV files (full and filtered)."""
    logger.info("Writing sgRNA output to %s.tsv", output_file)
    sgrnas.write_csv(output_file + ".tsv", separator='\t')

    sgrnas_filtered = sgrnas.filter(
        (pl.col("guide").cast(str) != "no_guides_found") &
        (pl.col("guide").cast(str) != "be_not_usable")
    )
    logger.info("Writing filtered sgRNA output to %s_filtered.tsv", output_file)
    sgrnas_filtered.write_csv(output_file + "_filtered.tsv", separator='\t')


def output_guides_sam(
    sam_df: pl.DataFrame,
    output_file: str,
    refgenome: str,
) -> None:
    """Convert *sam_df* to a sorted, indexed BAM file."""
    if sam_df.is_empty():
        logger.warning("No variant guides could be identified; no BAM file will be written")
        return

    sam_path = output_file + "_filtered.sam"
    unsorted_bam = output_file + "_filtered_unsorted.bam"
    sorted_bam = output_file + "_filtered.bam"

    logger.info("Writing guide alignment output to %s", sorted_bam)
    sam_df.write_csv(sam_path, separator='\t', include_header=False)
    pysam.view(sam_path, '-b', '-o', unsorted_bam, '-t', refgenome + '.fai',
               catch_stdout=False)
    os.remove(sam_path)
    pysam.sort('-o', sorted_bam, unsorted_bam)
    os.remove(unsorted_bam)
    pysam.index(sorted_bam)
    logger.info("Finished writing sorted and indexed BAM to %s", sorted_bam)


if __name__ == "__main__":
    pass
