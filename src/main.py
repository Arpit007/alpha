import argparse
import multiprocessing as mp
import sys

"""
Run Modes:
TEST_MODE:  For Testing
INTERACTIVE_MODE: For User Interaction
"""

if __name__ == "__main__":
	mp.freeze_support()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--genre', help = 'Genre Top N Mode', action = "store_true")
	parser.add_argument('-i', '--interactive', help = 'Interactive Mode', action = "store_true")
	parser.add_argument('-n', '--topn', help = 'Top N Mode', action = "store_true")
	parser.add_argument('-t', '--test', help = 'Test Mode', action = "store_true")
	args = parser.parse_args()
	
	app = None
	
	if args.interactive:
		from src.app import iMode
		
		app = iMode
	elif args.topn:
		from src.app import topN
		
		app = topN
	elif args.genre:
		from src.app import genreTopN
		
		app = genreTopN
	else:
		from src.app import testMode
		
		app = testMode
	
	# Run the App
	app.run()
	
	sys.exit(0)
