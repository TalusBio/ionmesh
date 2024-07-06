#!/bin/bash

# set -x # Display expansions
set -e
set -u
set -o pipefail

for x in *_tdf; do
	echo "Processing $x"
	dotd_name=$x/data.d
	tdf_name=$dotd_name/analysis.tdf
	tdf_bin_name=$dotd_name/analysis.tdf_bin

	# Check if the .d directory exists
	if [ -d $x/data.d ]; then
		echo "Directory $x/data.d exists"
		rm -rf $x/data.d
	fi

	echo "Creating $x/data.d does not exist"
	mkdir $x/data.d

    tdf_create=" \
    CREATE TABLE DiaFrameMsMsInfo ( \
        Frame INTEGER PRIMARY KEY, \
        WindowGroup INTEGER NOT NULL, \
        FOREIGN KEY (Frame) REFERENCES Frames (Id), \
        FOREIGN KEY (WindowGroup) REFERENCES DiaFrameMsMsWindowGroups (Id) \
    ); \
    CREATE TABLE DiaFrameMsMsWindowGroups ( \
        Id INTEGER PRIMARY KEY \
    ); \
    CREATE TABLE DiaFrameMsMsWindows ( \
        WindowGroup INTEGER NOT NULL, \
        ScanNumBegin INTEGER NOT NULL, \
        ScanNumEnd INTEGER NOT NULL, \
        IsolationMz REAL NOT NULL, \
        IsolationWidth REAL NOT NULL, \
        CollisionEnergy REAL NOT NULL, \
        PRIMARY KEY(WindowGroup, ScanNumBegin), \
        FOREIGN KEY (WindowGroup) REFERENCES DiaFrameMsMsWindowGroups (Id) \
    ) WITHOUT ROWID; \
    CREATE TABLE TimsCalibration ( \
        Id INTEGER PRIMARY KEY, \
        ModelType INTEGER NOT NULL, \
        C0 \
    , C1, C2, C3, C4, C5, C6, C7, C8, C9); \
    CREATE TABLE MzCalibration ( \
        Id INTEGER PRIMARY KEY, \
        ModelType INTEGER NOT NULL, \
        DigitizerTimebase REAL NOT NULL, \
        DigitizerDelay REAL NOT NULL, \
        T1 REAL NOT NULL, \
        T2 REAL NOT NULL, \
        dC1 REAL NOT NULL, \
        dC2 REAL NOT NULL, \
        C0 \
    , C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14); \
    CREATE TABLE Frames ( \
        Id INTEGER PRIMARY KEY, \
        Time REAL NOT NULL, \
        Polarity CHAR(1) CHECK (Polarity IN ('+', '-')) NOT NULL, \
        ScanMode INTEGER NOT NULL, \
        MsMsType INTEGER NOT NULL, \
        TimsId INTEGER, \
        MaxIntensity INTEGER NOT NULL, \
        SummedIntensities INTEGER NOT NULL, \
        NumScans INTEGER NOT NULL, \
        NumPeaks INTEGER NOT NULL, \
        MzCalibration INTEGER NOT NULL, \
        T1 REAL NOT NULL, \
        T2 REAL NOT NULL, \
        TimsCalibration INTEGER NOT NULL, \
        PropertyGroup INTEGER, \
        AccumulationTime REAL NOT NULL, \
        RampTime REAL NOT NULL, \
        Pressure REAL, \
        FOREIGN KEY (MzCalibration) REFERENCES MzCalibration (Id), \
        FOREIGN KEY (TimsCalibration) REFERENCES TimsCalibration (Id), \
        FOREIGN KEY (PropertyGroup) REFERENCES PropertyGroups (Id) \
    ); \
    CREATE TABLE GlobalMetadata ( \
        Key TEXT PRIMARY KEY, \
        Value TEXT \
    ); \
    "

    echo "Creating tables"
    sqlite3 ${tdf_name} "${tdf_create}"
    # Show schema
    sqlite3 ${tdf_name} ".schema"

	echo "DiaFrameMsMsWindowGroups >>>"
	sqlite3 -cmd ".mode csv" -separator $'\t' ${tdf_name} ".import --skip 1 ${x}/dia_frame_msms_window_groups.tsv DiaFrameMsMsWindowGroups"
	sqlite3 ${tdf_name} "SELECT * FROM DiaFrameMsMsWindowGroups LIMIT 5"
	echo "DiaFrameMsMsWindows >>>"
	sqlite3 -cmd ".mode csv"  -separator $'\t' ${tdf_name} ".import --skip 1 ${x}/dia_frame_msms_windows.tsv DiaFrameMsMsWindows"
	sqlite3 ${tdf_name} "SELECT * FROM DiaFrameMsMsWindows LIMIT 5"
	echo "DiaFrameMsMsInfo >>>"
	sqlite3 -cmd ".mode csv"  -separator $'\t' ${tdf_name} ".import --skip 1 ${x}/dia_frame_msms_info.tsv DiaFrameMsMsInfo"
	sqlite3 ${tdf_name} "SELECT * FROM DiaFrameMsMsInfo  LIMIT 5"
	echo "Frames >>>"
	sqlite3 -cmd ".mode csv"  -separator $'\t' ${tdf_name} ".import --skip 1 ${x}/frames.tsv Frames"
	sqlite3 ${tdf_name} "SELECT * FROM Frames LIMIT 5"
	echo "Global Metadata >>>"
	sqlite3 -cmd ".mode csv"  -separator $'\t' ${tdf_name} ".import --skip 1 ${x}/global_metadata.tsv GlobalMetadata"
	sqlite3 ${tdf_name} "SELECT * FROM GlobalMetadata LIMIT 5"

	echo "Creating tdf_bin"
	touch ${tdf_bin_name}
	# sqlite3 -separator ',' ${tdf_name} ".import ${x}/dia_frame_msms_window_groups.tsv DiaFrameMsMsWindowGroups"

done
