from sklearn.preprocessing import  LabelBinarizer

Ib = LabelBinarizer()

X = ['A', 'B', 'C', 'F', 'D', 'B']

Ib.fit(X)
print("어떻게 분류 했나 : ",Ib.classes_)
print("0과 1로 변형된 값 : ", Ib.transform(X))
