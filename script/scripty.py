from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import csv
from numpy.random import shuffle
from numpy.random import choice
import numpy
import os
import math


class CSVDataProcessor:

    def __init__(self, csvfile_filepath):
        self.filepath = csvfile_filepath
        self.data = self.load(self.filepath, duplicate=False)
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.division = 0.8

    def load(self, filepath, duplicate=True):
        """Load data from csv file"""
        data = []
        with open(filepath, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if int(row[-1]) == 2 and numpy.random.randint(0, 10) == 0:

                    data.append([int(el) for el in row])
                elif int(row[-1]) != 2:
                    data.append([int(el) for el in row])


        # duplicate settings and stage directions in order to better learn their embeddings
        # recommend: Do Not Use
        if duplicate:
            dups1 = []
            dups2 = []
            for row in data:
                if row[-1] == 0:
                    dups1.append(row)
                if row[-1] == 0:
                    dups2.append(row)
            for i in range(int(len(dups1)/2)):
                data.append(dups1[i])


        return data

    def examine(self):
        """Print the number of training examples with target class, index corresponds to class."""
        class_count = [0,0,0]
        for row in self.data:
            class_count[row[-1]] += 1
        print(class_count)

    def partition(self):
        # randomize data and pick training examples randomly
        shuffle(self.data)
        x_indexes = choice(range(len(self.data)), size=int(len(self.data)*self.division))

        #divide data into 4 sets for training
        for i in range(len(self.data)):
            if i in x_indexes:
                self.X_train.append(self.data[i][:-1])
                self.Y_train.append(self.data[i][-1])
            else:
                self.X_test.append(self.data[i][:-1])
                self.Y_test.append(self.data[i][-1])

    def debug(self):
        return


class ScriptSeqNet:

    def __init__(self, x_train, y_train, x_test, y_test, input_shape):
        self.input_shape = input_shape
        self.X_train = numpy.array(x_train)
        self.Y_train = numpy.array(y_train)
        self.X_test = numpy.array(x_test)
        self.Y_test = numpy.array(y_test)
        self.model = Sequential()
        self.mutation_rate = 4
        self.build_model()
        self.model.load_weights("models/model.h5")


    def build_model(self):
        print("Building model...")
        self.model.add(Dense(self.input_shape,activation="sigmoid", input_dim=self.input_shape))
        self.model.add(Dense(30, activation="sigmoid"))
        self.model.add(Dense(30, activation="sigmoid"))
        self.model.add(Dense(3,activation="softmax"))
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, model_name):
        self.model.summary()
        print("Training...")

        # one-hot encode the target set
        encoded_train = to_categorical(self.Y_train, num_classes=3)

        # fit model
        self.model.fit(self.X_train, encoded_train, epochs=200, batch_size=10)

        # evaluate model
        encoded_test = to_categorical(self.Y_test, num_classes=3)
        score = self.model.evaluate(self.X_test, encoded_test, batch_size=10)
        print("[Loss, Accuracy]:", score)

        self.model.save("models/" + model_name + ".h5")

    def load_model(self, filepath):
        self.model.load_weights(filepath)

    def find_start(self):
        """get row from the training data to use as a start point for generation"""
        for row in self.X_train:
            if row[-1] != 0:
                return row

    def generate(self, aslist=False):
        """mess"""
        elements = [0,1,2]
        # initialize script, convert to numpy.array (req by keras)
        script = numpy.array(self.find_start())

        # load model
        script_len = 400

        # generate
        for i in range(script_len):

            # predict the next script element based on the last 99
            pred = self.model.predict(numpy.array([script[i:i+self.input_shape]]))[0]
            if numpy.random.randint(0,100) >= self.mutation_rate:
                shuffle(pred)
            # pick new element based on returned softmax probabilities
            element = numpy.random.choice(elements, p=pred)

            # convert to list to add new element, convert back to numpy array
            script = list(script)
            script.append(element)
            script = numpy.array(script)

        if aslist:
            return list(script[self.input_shape:])

        else:
            return numpy.reshape(list(script[self.input_shape:]), (40,10))


class LineGenerator:

    def __init__(self, charpath, sdpath, setpath):
        self.chars = self.load(charpath)
        self.sds = self.load(sdpath)
        self.sds = [sd[0] for sd in self.sds]
        self.sets = self.load(setpath)
        self.sets = [sets[0] for sets in self.sets]
        self.chars_data = self.get_chars()

    def load(self, path):
        file = open(path, "r")
        data = []
        r = csv.reader(file, delimiter=",", quotechar="|")
        for row in r:
            if len(row) > 0:
                data.append(row)
        file.close()
        return data

    def get_chars(self):
        chars_data = {}

        for line in self.chars:
            chars_data[line[0]] = []

        for line in self.chars:
            chars_data[line[0]].append(line[1])
        return chars_data

    def realize(self, script):
        filled_script = []
        for id in script:
            if id == 0:
                key = numpy.random.choice(list(self.chars_data.keys()))
                filled_script.append(["char", key, numpy.random.choice(self.chars_data[key])])
            if id == 1:
                filled_script.append(["set", numpy.random.choice(self.sets)])
            if id == 2:
                filled_script.append(["sd", numpy.random.choice(self.sds)])

        return filled_script

    def debug(self):
        for el in self.chars:
            print(el)


class Generator:

    def __init__(self, seq_gen, line_gen):
        self.seq_gen = seq_gen
        self.line_gen = line_gen
        self.works = []
        self.realized_works = []
        self.examples = self.load_ex()

    def load_ex(self):
        ex = []
        for filename in os.listdir("script_arrays"):
            ex.append(numpy.genfromtxt("script_arrays/" + filename, delimiter=','))

        return ex

    def name(self):
        name = "The"
        places = ["House", "Store", "Cafe", "Shelf", "Bed", "Event", "Workplace", "Shop", "Pez", "Phone", "Raincoat",
                  "Secret", "Subway", "Wizard", "TV", "Computer", "Kramer", "George", "Jerry", "Elaine", "Apartment"]
        event = ["Catastrophe", "Debacle", "Challenge", "Argument", "Flirtation", "Crazyness", "Big-To-Do",
                 "Difficult", "Elf", "Book", "Lamp", "Bumble", "Roaster", "Comeback", "Conversion", "Girlfriend",
                 "Gum", "Biologist", "Garage"]
        name += " " + numpy.random.choice(places)
        name += " " + numpy.random.choice(event)
        return name

    def create(self, num_works):
        for i in range(num_works):
            self.works.append([self.name(), self.seq_gen.generate(), 0, 0])

        for work in self.works:
            self.realized_works.append([work[0], self.line_gen.realize(self.formated_to_list(work[1]))])

    def formated_to_list(self, work):
        lis = []
        for row in work:
            for id in row:
                lis.append(id)
        return lis

    def eval(self):
        # creativity score
        for work in self.works:
            for other_work in self.works:
                work[2] += self.dist(work[1], other_work[1])
            work[2] /= len(self.works)

        # domain competency score
        for work in self.works:
            total = 0
            for ex in self.examples:
                total += self.dist(work[1], ex, num_rows=10)
            total /= len(self.examples)
            work[3] = total

    def display(self):
        print("Keys: [originality, domain competency]")
        print("Originality = avg dist from own works")
        print("Domain Competency = avg dist from past works")
        for work in self.works:
            print(work[0], work[2:])
        return

    def dist(self, new_work, old_work, num_rows=40):
        disim_count = 0
        for i in range(num_rows):
            for j in range(len(new_work[i])):
                if new_work[i][j] != old_work[i][j]:
                    disim_count += 1
        return disim_count

    def works_to_txt(self):
        """Takes each realized work and outputs it as a formated script"""
        for work in self.realized_works:
            file = open("works/"+work[0] + ".txt", "w")
            script = work[1]
            for row in script:
                print(row)
                print(row[0])
                if row[0] == "char":

                    file.write((" " * 10) + row[1]+ "\n")
                    cnt = 0
                    for word in row[2].split():
                        if cnt >= 10:
                            cnt = 0
                            file.write("\n")
                            file.write(" " * 5)
                        file.write(" " + word)
                    file.write('\n\n')

                if row[0] == "set":
                    file.write("Setting: " + " " * 2 + row[1] + ")\n\n")
                if row[0] == "sd":
                    file.write("StageDir" + " " * 2 + row[1] + ")\n\n")
            file.close()

    def run_session(self):
        self.works = []
        self.realized_works = []
        print("-" *40)
        self.help()
        print("starting SCRIPTY session...")
        while True:
            command_line = input("<SCRIPTY>: ").split()
            print("thinking...")

            if len(command_line) > 2:
                print("invalid input -> use help command for API info ")
            elif command_line[0] == "generate":
                if len(command_line) != 2:
                    print("invalid input -> use help for info")
                else:
                    print("generating...", end="")
                    self.create(int(command_line[1]))
                    print("complete!")
                    print("evaluating...", end="")
                    self.eval()
                    print("complete!")
            elif command_line[0] == "write":
                print("writing...", end="")
                self.works_to_txt()
                print("complete!")
            elif command_line[0] == "train":
                if len(command_line) != 2:
                    print("invalid input -> use help for info")
                else:
                    print("training...", end="")
                    self.seq_gen.train(command_line[1])
                    print("complete!")
            elif command_line[0] == "load":
                if len(command_line) != 2:
                    print("invalid input -> use help for info")
                else:
                    print("loading...", end="")
                    self.seq_gen.load_model(command_line[1])
                    print("complete!")
            elif command_line[0] == "display":
                self.display()
            elif command_line[0] == "clear":
                os.system("rm works/*")
            elif command_line[0] == "list":
                os.system("ls works")
            elif command_line[0] == "help":
                self.help()
            elif command_line[0] == "quit":
                exit(0)
            else:
                print("invalid input -> use help command for API info ")


    def help(self):
        print("Welcome to SCRIPTY, an automatic script framework generator for Seinfeld.")
        print("Valid commands:")
        print("     - generate <num> # generate 'num' new sequences. Will overwrite past work")
        print("     - write # write all realized works to a file in the 'works' directory.")
        print("    - clear # rm all files from the works directory.")
        print("    - list # list all files in the works directory.")
        print("     - display # display all works and evaluate them.")
        print("     - help # display help message with API info.")
        print("     - train <model_name> # retrain model and save model to file with name model_name + .h5.")
        print("     - load <model_path> # load model at the specified filepath")
        print("     - quit # exit interface.")



def print_script(script):
    cnt = 0
    for el in script:
        print(el, end=",")
        if cnt >= 30:
            print()
            cnt = 0
        cnt += 1

if __name__ == "__main__":
    print("Loading data...")
    line_gen = LineGenerator("data/char_lines.csv", "data/sd_lines.csv", "data/set_lines.csv")
    cdp = CSVDataProcessor("data/script_seq_data.csv")
    cdp.partition()
    net = ScriptSeqNet(cdp.X_train, cdp.Y_train, cdp.X_test, cdp.Y_test, 99)
    gen = Generator(net, line_gen)
    gen.run_session()


