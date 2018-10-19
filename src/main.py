import multiprocessing as mp
import sys

"""
Run Modes:
Test Mode:  For Testing
Interactive Mode: For User Interaction
"""

APP_MODES = {
	'TEST_MODE': 1,
	'INTERACTIVE_MODE': 2
}

DEFAULT_MODE = APP_MODES['TEST_MODE']

if __name__ == "__main__":
	mp.freeze_support()
	
	appMode=DEFAULT_MODE
	#appMode = sys.argv[0] if len(sys.argv) > 0 else DEFAULT_MODE
	app = None
	
	if appMode == APP_MODES['TEST_MODE']:
		from src.app import testMode
		
		app = testMode
	elif appMode == APP_MODES['INTERACTIVE_MODE']:
		from src.app import iMode
		
		app = iMode
	
	if app is None:
		print("Invalid App Mode")
		sys.exit(-1)
	
	# Run the App Mode
	app.run()
	
	sys.exit(0)
