"""
Task:
(1) Inspect the Revelio ``company_ref`` table in Microsoft Fabric.

Workflow:
(1) Read the table and retain the validated columns of interest.
(2) Specify the unit of observation, identifiers, and variable meanings.
(3) Print the dataset documentation using the live Spark schema.
(4) Export the distinct combinations of the selected industry variables.
(5) Export a lightweight sample of 1,000 observations.

The exported value files describe values observed when this script is run.
They should not be interpreted as a permanent or theoretical Revelio taxonomy.

Wang Wenzhi, with the help of Codex
Time: 2026-07-14
"""

from pyspark.sql import SparkSession

"""
Notes:
(1) There is no need to add extension to the file name.
(2) Save the desired output as a Delta table and a Parquet dataset with the ``write`` method.
"""
DATASET_NAME = "company_ref"
OUTPUT_TABLE_NAME = "z_universe_industries"
OUTPUT_INDUSTRY_UNIVERSE = "Files/WenzhiW/List_StandardizedIndustries_InCompanyRef"
SAMPLE_SIZE = 1000
OUTPUT_SAMPLE_TABLE = "z_sample_company_ref"
OUTPUT_SAMPLE = "Files/WenzhiW/Sample_CompanyReference"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Read the company reference and select validated columns
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spark = SparkSession.builder.getOrCreate()
company_ref = spark.read.table(DATASET_NAME)
company_ref = company_ref.select(
    "rcid",
    "company",
    "primary_name",
    "factset_entity_id",
    "year_founded",
    "ticker",
    "exchange_name",
    "sedol",
    "isin",
    "cusip",
    "url",
    "naics_code",
    "cik",
    "lei",
    "linkedin_url",
    "child_rcid",
    "child_company",
    "child_linkedin_url",
    "ultimate_parent_rcid",
    "ultimate_parent_rcid_name",
    "gvkey",
    "ein",
    "hq_street_address",
    "hq_zip_code",
    "hq_city",
    "hq_metro_area",
    "hq_state",
    "hq_country",
    "hq_region",
    "rics_k50",
    "rics_k200",
    "rics_k400",
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Specify the dataset and variable documentation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


DATASET_SPEC = {
    "unit_of_observation": "One company covered by the delivered data",
    "candidate_key": ("rcid",),
    "linkage_identifiers": (
        "rcid",
        "ultimate_parent_rcid",
        "factset_entity_id",
        "gvkey",
        "ticker",
        "cusip",
        "isin",
        "lei",
    ),
}
"""
Notes:
(1) "rcid" is Revelio's documented company identifier.
(2) To keep this inspection lightweight, the script records "rcid" as the candidate key but does
    not run a computational uniqueness check.
(3) ``primary_name``, ``ein``, the headquarters fields, and the ``rics`` fields are present in the
    delivery but are not listed under Company Reference in either reference dictionary. Their
    labels therefore identify them as delivery-specific where appropriate.
"""

VARIABLE_MEANINGS = {
    "rcid": "Revelio Labs company ID.",
    "company": "Company name after Revelio Labs mapping.",
    "primary_name": "Delivery-specific primary or canonical company name.",
    "factset_entity_id": "FactSet company ID.",
    "year_founded": "Year in which the company was founded.",
    "ticker": "Ticker of the company.",
    "exchange_name": "Stock exchange on which the company is listed.",
    "sedol": "SEDOL code.",
    "isin": "ISIN code.",
    "cusip": "CUSIP number.",
    "url": "Company website URL.",
    "naics_code": "Six-digit NAICS industry code of the company.",
    "cik": "SEC Central Index Key number.",
    "lei": "Legal Entity Identifier code.",
    "linkedin_url": "Company LinkedIn URL.",
    "child_rcid": "Revelio Labs company ID of the largest subsidiary company.",
    "child_company": "Company name of the largest subsidiary company.",
    "child_linkedin_url": "Company LinkedIn URL of the largest subsidiary company.",
    "ultimate_parent_rcid": "Revelio Labs company ID of the ultimate parent company.",
    "ultimate_parent_rcid_name": "Revelio Labs company name of the ultimate parent company.",
    "gvkey": "GVKEY number of the company.",
    "ein": "Delivery-specific US Employer Identification Number of the company.",
    "hq_street_address": "Delivery-specific street address of company headquarters.",
    "hq_zip_code": "Delivery-specific postal code of company headquarters.",
    "hq_city": "Delivery-specific city of company headquarters.",
    "hq_metro_area": "Delivery-specific metropolitan area of company headquarters.",
    "hq_state": "Delivery-specific state or first-level administrative area of company headquarters.",
    "hq_country": "Delivery-specific country of company headquarters.",
    "hq_region": "Delivery-specific broad world region of company headquarters.",
    "rics_k50": "Revelio Labs industry classification of the company with 50 categories.",
    "rics_k200": "Revelio Labs industry classification of the company with 200 categories.",
    "rics_k400": "Revelio Labs industry classification of the company with 400 categories.",
}


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Print the dataset documentation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


print("\n" + "=" * 100)
print(f"Dataset name: {DATASET_NAME}  ")
print(f"Unit of observation: {DATASET_SPEC['unit_of_observation']}  ")
print(f"Candidate key: {' + '.join(DATASET_SPEC['candidate_key'])}  ")
print(
    f"Potential identifiers used to link with other datasets: {', '.join(DATASET_SPEC['linkage_identifiers'])}  "
)
print("Variable label and spark schema:  ")
for field in company_ref.schema.fields:
    print(
        f"  - {field.name}\n"
        f"    - Label: {VARIABLE_MEANINGS[field.name]}\n"
        f"    - Spark schema: [{field.dataType.simpleString()}; nullable={field.nullable}]\n"
    )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Export observed universe of industries
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INDUSTRY_VARIABLES = (
    "rics_k50",
    "rics_k200",
    "rics_k400",
    "naics_code",
)

"""
Notes:
(1) Keep only the industry classifications and remove repeated combinations.
(2) Keep all 4 variables to preserve the observed mapping between the "rics" and "naics" codes.
"""

industry_universe = company_ref.select(*INDUSTRY_VARIABLES).dropDuplicates()

(
    industry_universe.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE_NAME)
)
print(f"Table saved: {OUTPUT_TABLE_NAME}.")

(
    spark.read.table(OUTPUT_TABLE_NAME)
    .coalesce(1)
    .write.mode("overwrite")
    .parquet(OUTPUT_INDUSTRY_UNIVERSE)
)
print(f"Parquet dataset saved: {OUTPUT_INDUSTRY_UNIVERSE}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Export a lightweight sample of observations
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


sample_data = company_ref.limit(SAMPLE_SIZE)

(
    sample_data.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_SAMPLE_TABLE)
)
print(f"Table saved: {OUTPUT_SAMPLE_TABLE}.")

(spark.read.table(OUTPUT_SAMPLE_TABLE).coalesce(1).write.mode("overwrite").parquet(OUTPUT_SAMPLE))
print(f"Parquet dataset saved: {OUTPUT_SAMPLE}.")
