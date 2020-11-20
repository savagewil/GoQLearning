from DataProcessing.SGF import SGF

FILE  = open("C:\\Users\\wills\\Code\\Python3\\GoQLearning\\Data\\games\\games\\Aizu\\01\\1.sgf","r")
SDF_Data = "".join(FILE.readlines())
FILE.close()

SGF_Data = SGF.FromString(SDF_Data)

print(SGF_Data.Event)
