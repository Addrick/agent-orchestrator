import logging
from typing import Optional, Tuple, Dict, Set, Any, List

from google.genai.types import GroundingMetadata


def process_grounding_metadata(base_text_from_response: str, metadata: Optional[GroundingMetadata],
                               logger: logging.Logger) -> Tuple[str, str, str]:
    """Processes grounding metadata to insert citations and list sources."""
    if not (metadata and metadata.grounding_chunks and metadata.grounding_supports):
        return base_text_from_response, "", ""

    insertion_points_map: Dict[int, Set[int]] = {}
    current_search_cursor: int = 0

    processed_sources: Dict[str, Dict[str, Any]] = {}
    source_id_counter: int = 1
    chunk_idx_to_source_id_map: Dict[int, int] = {}

    for chunk_index, chunk in enumerate(metadata.grounding_chunks):
        if chunk.web and chunk.web.uri:
            uri: str = chunk.web.uri
            title: str = chunk.web.title or uri
            if uri not in processed_sources:
                processed_sources[uri] = {'id': source_id_counter, 'title': title, 'url': uri}
                source_id_counter += 1
            chunk_idx_to_source_id_map[chunk_index] = processed_sources[uri]['id']

    segments_to_cite: List[Dict[str, Any]] = []
    for support in metadata.grounding_supports:
        s_text: str = support.segment.text
        s_start_hint: int = support.segment.start_index if support.segment.start_index is not None else 0

        supporting_source_ids: Set[int] = set()
        if support.grounding_chunk_indices:
            for c_idx in support.grounding_chunk_indices:
                if c_idx in chunk_idx_to_source_id_map:
                    supporting_source_ids.add(chunk_idx_to_source_id_map[c_idx])

        if s_text and supporting_source_ids:
            segments_to_cite.append({
                "start_hint": s_start_hint,
                "text": s_text,
                "citations": sorted(list(supporting_source_ids))
            })

    segments_to_cite.sort(key=lambda s: s["start_hint"])

    for segment_data in segments_to_cite:
        segment_text: str = segment_data["text"]
        citation_ids_for_segment: List[int] = segment_data["citations"]

        found_index: int = base_text_from_response.find(segment_text, current_search_cursor)
        if found_index == -1:
            alt_found_index: int = base_text_from_response.find(segment_text, 0)
            if alt_found_index != -1:
                found_index = alt_found_index
            else:
                continue

        insertion_location: int = found_index + len(segment_text)

        if citation_ids_for_segment:
            if insertion_location not in insertion_points_map:
                insertion_points_map[insertion_location] = set()
            insertion_points_map[insertion_location].update(citation_ids_for_segment)

        current_search_cursor = max(current_search_cursor, insertion_location)

    text_with_citations: str = base_text_from_response
    if insertion_points_map:
        modified_text_parts: List[str] = []
        last_slice_end: int = 0
        for loc in sorted(insertion_points_map.keys()):
            modified_text_parts.append(base_text_from_response[last_slice_end:loc])

            hyperlinked_citation_parts: List[str] = []
            sorted_citation_ids_at_loc: List[int] = sorted(list(insertion_points_map[loc]))

            for src_id in sorted_citation_ids_at_loc:
                source_info: Optional[Dict[str, Any]] = next(
                    (s_info for s_info in processed_sources.values() if s_info['id'] == src_id),
                    None)
                if source_info:
                    hyperlinked_citation_parts.append(f"[{src_id}](<{source_info['url']}>)")

            citation_str_to_insert: str = ""
            if hyperlinked_citation_parts:
                citation_str_to_insert = f" [{', '.join(hyperlinked_citation_parts)}]"

            modified_text_parts.append(citation_str_to_insert)
            last_slice_end = loc
        modified_text_parts.append(base_text_from_response[last_slice_end:])
        text_with_citations = "".join(modified_text_parts)

    citations_text_list_str: str = ""
    if processed_sources:
        citations_text_list_str = "\n\nSources:\n"
        ordered_sources_for_display: List[Dict[str, Any]] = sorted(processed_sources.values(), key=lambda s: s['id'])
        for src_info in ordered_sources_for_display:
            citations_text_list_str += f"{src_info['id']}. {src_info['title']}\n"

    search_query_text_str: str = ""
    if metadata.web_search_queries:
        search_query_text_str = f"\n\nSearch Query: {', '.join(metadata.web_search_queries)}"

    return text_with_citations, search_query_text_str, citations_text_list_str
