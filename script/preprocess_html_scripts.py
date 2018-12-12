import xml.etree.ElementTree as ET
import os
def isSetting(text):
	text = text.strip()
	if text == "":
		return False
	return text[0] == "(" and text[-1] == ")"


def preprocess_html(filename):
	html_et = ET.parse(filename)
	# b tags contain the script information
	b_tags = html_et.iter(tag='b')

	# create new tree
	root = ET.Element("root")

	for tag in b_tags:

		# if tag contains text it is a Character tag.
		if tag.text.strip() != "":

			# build a new element with attribute of chars name and add to new html
			char_el = ET.Element("char")
			char_el.set("name",tag.text.strip())
			char_el.set("dialog", "")
			root.append(char_el)

			# check if tail of tag contains text, if so it is dialogue, add it as an attribute
			if tag.tail != None and tag.text.strip() != "":
				char_el.set("dialog", " ".join(tag.tail.strip().split()))

		# if the tag has a tail and no text it is a set or sd
		if tag.tail != None and tag.text.strip() == "":
			if isSetting(tag.tail.strip()):
				el = ET.Element("set")

			# occasionally mislabels dialogue as a stage direction
			else:
				el = ET.Element("sd")
			el.text = " ".join(tag.tail.strip().split())
			root.append(el)

	processed_html_et = ET.ElementTree(root)

	processed_html_et.write("processed_html_scripts/%s" % filename.replace("html_scripts/",""))


if __name__ == "__main__":
	for filename in os.listdir("html_scripts"):
		preprocess_html("html_scripts/" + filename)

