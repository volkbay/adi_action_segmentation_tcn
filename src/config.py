
###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

# Volkan Okbay (2022)
# Config YAML file parser and argument class for ai8x normalization

import yaml

class ConfigParser:
    def __init__(self, config_file="config.yaml"):
        self.f = config_file

    def read(self, word):
        with open(self.f, 'r') as stream:
            dictionary = yaml.load(stream, Loader=yaml.FullLoader)        
            return dictionary[word]

    def readAll(self):
        with open(self.f, 'r') as stream:
            dictionary = yaml.load(stream, Loader=yaml.FullLoader)        
            return dictionary

    def write(self, word, val):
        with open(self.f, 'r') as stream:
            dictionary = yaml.load(stream, Loader=yaml.FullLoader)        
        dictionary[word] = val
        with open(self.f, 'w') as stream:
            yaml.dump(dictionary, stream)
        
    def printAll(self):
        with open(self.f, 'r') as stream:
            dictionary = yaml.load(stream, Loader=yaml.FullLoader)
        print("I - CONFIGURATION:", dictionary)
        return

class Args:
    def __init__(self, act_mode_8bit):
        self.act_mode_8bit = act_mode_8bit