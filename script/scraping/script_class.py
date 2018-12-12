import parse_html_script
class Character():

	def __init__(self, name):
		self.name = name
		self.dialog = ""

class Setting():

	def __init__(self, text):
		self.text = text

class StageDir():

	def __init__(self, text):
		self.text = text

class Script():
	
	def __init__(self, name):
		self.name = name
		self.characters = []
		self.char_dialog = {}
		self.stage_directions = []
		self.script = []

	def load_script_from_html(self, filepath):
		self.name = filepath
		self.script = self.load()

