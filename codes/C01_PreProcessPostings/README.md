## Terminology in the codes

1. Postings: Job postings, identified by the raw `job_id` variable.
2. Descriptions: Job description texts of "postings". 
   1. They could be empty, meaningless, or duplicated. 
   2. They could have HTML tags used to infer the structure of the description text and decompose the text into several blocks.
3. Normalized: Normalized job description texts.
   1. From "descriptions" to "normalized" (i.e., the normalization process), the biggest change is that HTML tags are removed, while the original structure is preserved separately in blocks.
4. Blocks: Minimal identifiable unit in "normalized" descriptions.
   1. During the normalization process, information in the HTML tags is preserved as different blocks within one "normalized" description.
   2. There are 5 block types: "HEADING", "LIST_ITEM", "PARAGRAPH", "SOURCE_LINE", and "TABLE_ROW".
   3. Blocks for a "normalized" description is stored as a string of json format with the following fields: "block_type", "heading_level", and "text".





