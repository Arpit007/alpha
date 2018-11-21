import argparse
import multiprocessing as mp
import sys

"""
Run Modes:
GENRE_MODE: Top N Movies of Given Genre(s)
INTERACTIVE_MODE: For User Interaction
TOPN_MODE: Top N Movies
TEST_MODE:  For Testing
"""

if __name__ == "__main__":
	mp.freeze_support()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--categorise', help = 'Genre Top N Mode', action = "store_true")
	parser.add_argument('-g', '--gui', help = 'GUI Based Interaction', action = "store_true")
	parser.add_argument('-i', '--interactive', help = 'Interactive Mode', action = "store_true")
	parser.add_argument('-n', '--topn', help = 'Top N Mode', action = "store_true")
	parser.add_argument('-t', '--test', help = 'Test Mode', action = "store_true")
	args = parser.parse_args()
	
	app = None
	
	if args.interactive:
		from src.app import iMode as app
	elif args.topn:
		from src.app import topN as app
	elif args.categorise:
		from src.app import genreTopN as app
	elif args.gui:
		from src.app import guiMode as app
	else:
		from src.app import testMode as app
	
	# Run the App
	app.run()
	
	sys.exit(0)
