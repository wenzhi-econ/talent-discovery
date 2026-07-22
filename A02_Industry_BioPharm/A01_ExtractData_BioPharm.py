"""
Task:
    Extract biotechnology and pharmaceutical companies, employment spells at those companies,
    and the companies' LinkedIn and Indeed postings.

Inputs:
    company_ref
    user_positions
    postings_linkedin
    postings_indeed

Outputs:
    Files/WenzhiW/BioPharm/Companies
    Files/WenzhiW/BioPharm/UserPositions_FocalSpells
    Files/WenzhiW/BioPharm/LinkedInPostings
    Files/WenzhiW/BioPharm/IndeedPostings

Notes:
(1) Focal companies have one of three specified ``rics_k400`` values in ``company_ref``.
(2) All links use the focal company's ``rcid``; parent and subsidiary IDs are not substituted.
(3) Only employment spells whose ``rcid`` identifies a focal company are exported.
(4) Other employment spells of the same users are not retained.
(5) LinkedIn and Indeed posting records are filtered and exported separately.

Wang Wenzhi, with the help of Codex
Time: 2026-07-20
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


"""
Notes:
(1) Input names refer to tables in the Fabric Lakehouse attached to the notebook.
(2) Each output is saved both as a managed Delta table and as downloadable Parquet files.
(3) Outputs are coalesced to one Parquet part to make them easier to download.
"""
COMPANY_TABLE = "company_ref"
POSITION_TABLE = "user_positions"
LINKEDIN_TABLE = "postings_linkedin"
INDEED_TABLE = "postings_indeed"

FOCAL_INDUSTRIES = (
    "Biotechnology and Life Sciences",
    "Pharmaceutical Manufacturing",
    "Pharmaceuticals",
)
TABLE_COMPANIES = "z_biopharm_companies"
TABLE_POSITIONS = "z_biopharm_user_positions_focal_spells"
TABLE_LINKEDIN_POSTINGS = "z_biopharm_linkedin_postings"
TABLE_INDEED_POSTINGS = "z_biopharm_indeed_postings"

OUTPUT_COMPANIES = "Files/WenzhiW/BioPharm/Companies"
OUTPUT_POSITIONS = "Files/WenzhiW/BioPharm/UserPositions_FocalSpells"
OUTPUT_LINKEDIN_POSTINGS = "Files/WenzhiW/BioPharm/LinkedInPostings"
OUTPUT_INDEED_POSTINGS = "Files/WenzhiW/BioPharm/IndeedPostings"


def export_dataframe(
    dataframe: DataFrame,
    table_name: str,
    parquet_path: str,
    single_parquet_part: bool = False,
) -> None:
    """Save one DataFrame as a managed Delta table and a Parquet dataset."""
    (
        dataframe.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )

    parquet_dataframe = spark.read.table(table_name)
    if single_parquet_part:
        parquet_dataframe = parquet_dataframe.coalesce(1)

    parquet_dataframe.write.mode("overwrite").parquet(parquet_path)
    print(f"Delta table saved: {table_name}.")
    print(f"Parquet dataset saved: {parquet_path}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Select companies in the focal biotechnology and pharmaceutical industries
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spark = SparkSession.builder.getOrCreate()
company_ref = spark.read.table(COMPANY_TABLE)

"""
Notes:
(1) The filter uses exact values requested for the ``rics_k400`` variable.
(2) ``dropDuplicates`` protects later joins from multiplying records if an ``rcid`` is repeated.
(3) All company-reference variables are retained for flexible analysis after download.
"""
suitable_companies = (
    company_ref.filter(F.col("rics_k400").isin(*FOCAL_INDUSTRIES)).dropDuplicates(["rcid"]).cache()
)
print(f"Total number of focal companies: {suitable_companies.count():,}")
export_dataframe(suitable_companies, TABLE_COMPANIES, OUTPUT_COMPANIES, True)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Extract employment spells at focal companies
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>

"""
Notes:
(1) The small focal-company ID table is broadcast to avoid shuffling the large position table.
(2) A left-semi join retains a position if its ``rcid`` occurs in the focal-company ID table.
(3) Unlike an inner join, the left-semi join adds no company-reference variables or duplicate rows.
(4) All original position columns are retained, but positions at other companies are excluded.
(5) No date restriction is imposed on employment spells at focal companies.
"""
focal_company_ids = suitable_companies.select("rcid")

positions_in_focal_companies = (
    spark.read.table(POSITION_TABLE)
    .join(
        F.broadcast(focal_company_ids),
        on="rcid",
        how="left_semi",
    )
    .cache()
)

print(f"Employment spells at focal companies: {positions_in_focal_companies.count():,}")
export_dataframe(
    positions_in_focal_companies,
    TABLE_POSITIONS,
    OUTPUT_POSITIONS,
    True,
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Extract postings by focal companies separately for each source
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) Each posting source is filtered before it is saved, and the two sources are never appended.
(2) The small focal-company ID table is broadcast to avoid shuffling the large posting table.
(3) A left-semi join filters posting rows without adding duplicate company-reference variables.
(4) All original posting columns are retained for analysis on the local laptop.
(5) Posting records are not deduplicated or date-filtered during this extraction step.
"""


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-1. Extract LinkedIn postings by focal companies
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


linkedin_postings_in_focal_companies = spark.read.table(LINKEDIN_TABLE).join(
    F.broadcast(focal_company_ids),
    on="rcid",
    how="left_semi",
)

export_dataframe(
    linkedin_postings_in_focal_companies,
    TABLE_LINKEDIN_POSTINGS,
    OUTPUT_LINKEDIN_POSTINGS,
    True,
)


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-2. Extract Indeed postings by focal companies
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


indeed_postings_in_focal_companies = spark.read.table(INDEED_TABLE).join(
    F.broadcast(focal_company_ids),
    on="rcid",
    how="left_semi",
)

export_dataframe(
    indeed_postings_in_focal_companies,
    TABLE_INDEED_POSTINGS,
    OUTPUT_INDEED_POSTINGS,
    True,
)

positions_in_focal_companies.unpersist()
suitable_companies.unpersist()
