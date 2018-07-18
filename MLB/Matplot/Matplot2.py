import matplotlib.pyplot as plt

# x좌표
x_value = [1, 2, 3, 4, 5]

# y좌표
y_value = [1, 4, 9, 16, 25]

plt.plot(x_value, y_value, linewidth = 5)
plt.title("Get Square", fontsize = 20)
plt.xlabel("X value", fontsize = 20)
plt.ylabel("Y value", fontsize = 20)
plt.tick_params(axis = 'both', labelsize = 15)

plt.show()