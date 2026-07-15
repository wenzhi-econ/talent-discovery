"""
Task:
(1) Inspect the Revelio ``user_positions`` table in Microsoft Fabric.

Workflow:
(1) Read the table and retain the validated columns of interest.
(2) Specify the unit of observation, identifiers, and variable meanings.
(3) Print the dataset documentation using the live Spark schema.
(4) Export distinct combinations of the selected role and geography variables.
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
(2) Save the desired outputs as Delta tables and Parquet datasets with the ``write`` method.
"""
DATASET_NAME = "user_positions"

TABLE_ROLE_UNIVERSE = "z_universe_roles"
TABLE_GEOGRAPHY_UNIVERSE = "z_universe_geography"
OUTPUT_ROLE_UNIVERSE = "Files/WenzhiW/List_StandardizedRoles_InUserPositions"
OUTPUT_GEOGRAPHY_UNIVERSE = "Files/WenzhiW/List_Geography_InUserPositions"
SAMPLE_SIZE = 1000
OUTPUT_SAMPLE_TABLE = "z_sample_user_positions"
OUTPUT_SAMPLE = "Files/WenzhiW/Sample_UserPositions"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Read the position data and select validated columns
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spark = SparkSession.builder.getOrCreate()
user_positions = spark.read.table(DATASET_NAME)
user_positions = user_positions.select(
    "user_id",
    "position_id",
    "company_raw",
    "company_linkedin_url",
    "company_cleaned",
    "rcid",
    "company_name",
    "location_raw",
    "city",
    "msa",
    "metro_area",
    "state",
    "country",
    "region",
    "startdate",
    "enddate",
    "title_raw",
    "job_category",
    "role_k50",
    "role_k150",
    "role_k300",
    "role_k500",
    "role_k1000",
    "role_k1500",
    "remote_suitability",
    "weight",
    "description",
    "start_salary",
    "end_salary",
    "salary",
    "seniority",
    "position_number",
    "ultimate_parent_rcid",
    "ultimate_parent_company_name",
    "onet_code",
    "onet_title",
    "ticker",
    "exchange",
    "cusip",
    "naics_code",
    "naics_description",
    "rics_k50",
    "rics_k200",
    "rics_k400",
    "ultimate_parent_factset_id",
    "ultimate_parent_factset_name",
    "total_compensation",
    "additional_compensation",
    "title_translated",
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Specify the dataset and variable documentation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


DATASET_SPEC = {
    "unit_of_observation": "One position (employment spell) held by one individual",
    "candidate_key": ("position_id",),
    "linkage_identifiers": (
        "position_id",
        "user_id",
        "rcid",
        "ultimate_parent_rcid",
    ),
}
"""
Notes:
(1) "position_id" is Revelio's documented position-record identifier.
(2) To keep this inspection lightweight, the script records "position_id" as the candidate key but
    does not run a computational uniqueness check.
(3) ``msa`` and ``remote_suitability`` are present in the delivery but are absent from both
    reference dictionaries, so their labels are explicitly identified as delivery-specific.
"""

VARIABLE_MEANINGS = {
    "user_id": "Revelio Labs identifier for an individual public profile.",
    "position_id": "Revelio Labs identifier for a position or employment spell.",
    "company_raw": "Employer name as reported in the public profile.",
    "company_linkedin_url": "Employer LinkedIn URL reported in the public profile.",
    "company_cleaned": "Reported employer name cleaned of special characters.",
    "rcid": "Revelio Labs identifier for the mapped company.",
    "company_name": "Employer name after Revelio company mapping.",
    "location_raw": (
        "Raw position location from the public profile; if missing, inferred from the user's "
        "reported location."
    ),
    "city": "City of the position, imputed from the raw location.",
    "msa": "Delivery-specific Metropolitan Statistical Area label supplied for the position.",
    "metro_area": "Metropolitan area of the position, imputed from the raw location.",
    "state": "State or first-level administrative area of the position, imputed from the raw location.",
    "country": "Country of the position, imputed from the raw location.",
    "region": "Broad world region of the position, imputed from the raw location.",
    "startdate": "Position start date if reported; null otherwise.",
    "enddate": "Position end date if reported; null otherwise.",
    "title_raw": "Position title as reported in the public profile.",
    "job_category": "Mapped position role with 7 discrete levels.",
    "role_k50": "Mapped position role with 50 discrete levels.",
    "role_k150": "Mapped position role with 150 discrete levels.",
    "role_k300": "Mapped position role with 300 discrete levels.",
    "role_k500": "Mapped position role with 500 discrete levels.",
    "role_k1000": "Mapped position role with 1,000 discrete levels.",
    "role_k1500": "Mapped position role with 1,500 discrete levels.",
    "remote_suitability": (
        "Delivery-specific remote-suitability measure; its scale is not defined in either "
        "reference dictionary."
    ),
    "weight": (
        "Predicted sampling weight of the user, constructed to adjust occupation- and "
        "country-specific representation."
    ),
    "description": "Raw position description entered in the public profile.",
    "start_salary": "Modeled annual salary at the start of the position, in USD.",
    "end_salary": "Modeled annual salary at the end of the position, in USD.",
    "salary": "Modeled annual salary for the position, in USD.",
    "seniority": "Predicted ordinal seniority level, from 1 (entry) to 7 (senior executive).",
    "position_number": "Chronological order of the position within the user's profile.",
    "ultimate_parent_rcid": "Revelio identifier for the ultimate parent company.",
    "ultimate_parent_company_name": "Name of the ultimate parent company.",
    "onet_code": "Eight-digit predicted O*NET code of the position, mapped at the role_k1500 level.",
    "onet_title": "O*NET title of the position, mapped at the role_k1500 level.",
    "ticker": "Public-market ticker associated with the mapped company.",
    "exchange": "Stock exchange associated with the mapped company.",
    "cusip": "CUSIP security identifier associated with the mapped company.",
    "naics_code": "Six-digit NAICS industry code of the mapped company.",
    "naics_description": "Description of the mapped company's NAICS code.",
    "rics_k50": "Revelio Labs industry classification of the company with 50 categories.",
    "rics_k200": "Revelio Labs industry classification of the company with 200 categories.",
    "rics_k400": "Revelio Labs industry classification of the company with 400 categories.",
    "ultimate_parent_factset_id": "FactSet identifier for the ultimate parent company.",
    "ultimate_parent_factset_name": "FactSet name for the ultimate parent company.",
    "total_compensation": (
        "Modeled annual total compensation, including bonuses and benefits, for the position in USD."
    ),
    "additional_compensation": (
        "Modeled annual additional compensation, excluding base salary, for the position in USD."
    ),
    "title_translated": "Raw position title translated into English.",
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
for field in user_positions.schema.fields:
    print(
        f"  - {field.name}\n"
        f"    - Label: {VARIABLE_MEANINGS[field.name]}\n"
        f"    - Spark schema: [{field.dataType.simpleString()}; nullable={field.nullable}]\n"
    )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Export observed universes of selected categorical variables
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-4-1. The universe of user roles
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

ROLE_VARIABLES = (
    "job_category",
    "role_k50",
    "role_k150",
    "role_k300",
    "role_k500",
    "role_k1000",
    "role_k1500",
)

role_universe = user_positions.select(*ROLE_VARIABLES).dropDuplicates()

(
    role_universe.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_ROLE_UNIVERSE)
)
print(f"Table saved: {TABLE_ROLE_UNIVERSE}.")

(
    spark.read.table(TABLE_ROLE_UNIVERSE)
    .coalesce(1)
    .write.mode("overwrite")
    .parquet(OUTPUT_ROLE_UNIVERSE)
)
print(f"Parquet dataset saved: {OUTPUT_ROLE_UNIVERSE}.")

# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-4-2. The universe of geography locations
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

GEOGRAPHY_VARIABLES = (
    "country",
    "state",
    "city",
)

geography_universe = user_positions.select(*GEOGRAPHY_VARIABLES).dropDuplicates()

(
    geography_universe.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_GEOGRAPHY_UNIVERSE)
)
print(f"Table saved: {TABLE_GEOGRAPHY_UNIVERSE}.")

(
    spark.read.table(TABLE_GEOGRAPHY_UNIVERSE)
    .coalesce(1)
    .write.mode("overwrite")
    .parquet(OUTPUT_GEOGRAPHY_UNIVERSE)
)
print(f"Parquet dataset saved: {OUTPUT_GEOGRAPHY_UNIVERSE}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Export a lightweight sample of observations
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


sample_data = user_positions.limit(SAMPLE_SIZE)

(
    sample_data.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_SAMPLE_TABLE)
)
print(f"Table saved: {OUTPUT_SAMPLE_TABLE}.")

(spark.read.table(OUTPUT_SAMPLE_TABLE).coalesce(1).write.mode("overwrite").parquet(OUTPUT_SAMPLE))
print(f"Parquet dataset saved: {OUTPUT_SAMPLE}.")
