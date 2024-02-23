import glob
import gzip
from argparse import ArgumentParser

def create_offset_files(globstring, debug):
	print(globstring)
	for file in glob.glob(globstring):
		# outname = file.replace(".json", ".offsets").replace(".gz", "").replace("scott","xichen/mediacloud/ner_art_sampling")
		outname=file.replace(".json",".offsets").replace(".gz","")
		print(outname)

		if '.gz' not in file:
			with open(file, "r", ) as fhIn, open(outname, "w") as fhOut:
				offset = 0
				for line in fhIn:
					fhOut.write(f"{offset}\n")
					offset += len(line)
		else:
			with gzip.open(file, "rt") as fhIn, open(outname, "w") as fhOut:
				offset = 0
				for line in fhIn:
					fhOut.write(f"{offset}\n")
					offset += len(line)
		# try:
		# 	with open(file,"r",) as fhIn, open(outname,"w") as fhOut:
		# 		offset=0
		# 		for line in fhIn:
		# 			fhOut.write(f"{offset}\n")
		# 			offset+=len(line)
		# except UnicodeDecodeError:
		# 	with open(file,"r", encoding='ISO-8859-1') as fhIn, open(outname,"w") as fhOut:
		# 		offset=0
		# 		for line in fhIn:
		# 			fhOut.write(f"{offset}\n")
		# 			offset+=len(line)
	print("create offset files successfully...", flush=True)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-files", dest="input",
		default="/home/scott/ner/en/*.json", type=str,
		help="Path to input files.")
	parser.add_argument("-d", "--debug", dest="debug",
		default=False, action='store_true',
		help="Debug mode running for only a small chunk of data.")
	args = parser.parse_args()

	create_offset_files( args.input, args.debug )


