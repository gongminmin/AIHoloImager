# Copyright (c) 2025 Minmin Gong
#

import struct
import sys

def ParseDatabase(file_path):
    database = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line and (not line[0] == "#"):
                    splitted = line.split(";")
                    if len(splitted) == 2:
                        model, sensor_size = splitted
                        database[model] = float(sensor_size)
        return database
    except ValueError as e:
        print(f"Error: Invalid data format in file. {e}")
        return {}
    except Exception as e:
        print(f"Error: An unexpected error occurred. {e}")
        return {}

def WriteToBinary(database, file_path):
    try:
        with open(file_path, "wb") as file:
            file.write(struct.pack("<i", len(database)))
            for model, sensor_size in database.items():
                model = model.encode("ascii")
                file.write(struct.pack(f"<H{len(model)}sf", len(model), model, sensor_size))
        print(f"Successfully wrote to binary file '{file_path}'")
    except Exception as e:
        print(f"Error writing to binary file: {e}")

if __name__ == "__main__":
    database = ParseDatabase(sys.argv[1])
    WriteToBinary(database, sys.argv[2])
