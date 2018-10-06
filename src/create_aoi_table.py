"""Create a new table in the OSM PostGIS database with the
areas of interest of each case study.
"""

import os
from shapely.geometry import shape
import psycopg2

from metadata import CASE_STUDIES

if __name__ == '__main__':

    conn = psycopg2.connect(
        database='osm',
        user='maupp',
        password='maupp',
        host='localhost'
    )

    curs = conn.cursor()

    curs.execute("DROP TABLE IF EXISTS datafusion")
    curs.execute("CREATE TABLE datafusion (name TEXT)")
    curs.execute("SELECT AddGeometryColumn('datafusion', 'geom', 4326, 'POLYGON', 2)")

    for case_study in CASE_STUDIES:
        aoi = shape(case_study.aoi['geometry'])
        query = f"INSERT INTO datafusion (name, geom) VALUES ('{case_study.id}', ST_GeomFromWKB('{aoi.wkb_hex}'::geometry, 4326))"
        curs.execute(query)

    conn.commit()
    conn.close()
