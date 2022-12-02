import matplotlib.pyplot as plt


def getXY(filename):
    Xs = []
    Ys = []
    with open(filename) as f:
        lines = f.readlines()
        for line_i, line in enumerate(lines):
            if line_i + 1 <= 2019:
                continue
            tokens = line.split(' ')
            if tokens[0] == 'INFO:root:Dev':
                episode = int(tokens[2])
                if episode >= 7000:
                    break
                acc = float(tokens[-1])
                Xs.append(episode)
                Ys.append(acc)
    return Xs, Ys

def getTrainLoss(filename):
    Xs = []
    Ys = []
    k = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(' ')
            if 'Loss:' in tokens:
                loss = float(tokens[4])
                episode = int(tokens[2])
                Xs.append(episode)
                Ys.append(loss)
    return Xs, Ys



newX, newY = getXY('./3-16_seq_nlp.txt')
# oldX, oldY = getDiff('./experiment/2-26-test_diff_level2.txt')
# oldX = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 50000, 65000, 70000, 75000]
# oldY = [0.379, 0.417, 0.424, 0.456, 0.458, 0.469, 0.489, 0.491, 0.495, 0.503, 0.506, 0.514]
# newX = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]
# newY = [0.31, 0.32, 0.329, 0.362, 0.368, 0.381, 0.389, 0.391, 0.511]
# newY = [0.31, 0.32, 0.329, 0.362, 0.368, 0.381, 0.389, 0.391, 0.511]
# print(oldX, oldY)
# exit()
# newX, newY = getTrainLoss('./experiment/2-26-seq3.txt')
oldX, oldY = getXY('./3-16_non_seq_nlp.txt')
# oldX.append(newX[-1])
# oldY.append(oldY[-1])
newX.append(oldX[-1])
newY.append(newY[-1])
# print(newX, newY)
# exit()
plt.title("Validation Accuracy over Episode")
plt.xlabel("Episode")
plt.ylabel("Validation Accuracy")
plt.plot(newX,newY, color='turquoise', marker="x", label='Curriculum Learning')
plt.plot(oldX,oldY, color='grey', marker="x", label='Without Curriculum Learning')
plt.legend()
# plt.title('Training accuracy vs difficulty level')
# plt.ylabel('training accuracy')
# plt.xlabel('difficulty level')
plt.savefig('./3-16_val_acc.png')
# plt.show()