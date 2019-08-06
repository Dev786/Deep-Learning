import numpy as np

# set the equation for
# y = mx+c
m = 10
c = 5
total_test_points = 5000
batches = 1000
# using MSE
# J(w,b) = sum((y-y_predicted)**2)/total_test_points


class batch_gradient_descent():

    def __init__(self, epochs :int, weight : float, bias: float, learning_rate=0.01, x=None, y=None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        if x is None:
            self.x = np.linspace(1, 5, total_test_points)
        else:
            self.x = x

        self.y = self.x*m + c

        # setting initial variables
        self.w = weight
        self.b = bias

    def error(self, y_true, y_predicted):
        return np.square(y_true - y_predicted)

    def grad_w(self, x,y, y_predicted):
        return (-2 * x * (y-y_predicted))

    def grad_b(self, y,y_predicted):
        return (-2 * (y-y_predicted))

    def gradient_descent(self):
        loss = []
        biases = []
        weights = []
        total_inputs = len(self.x)

        for _ in range(self.epochs):
            dw = 0
            db = 0
            total_error = 0
            batch_indices = np.random.choice(total_test_points,batches)
            for x, y in zip(self.x[batch_indices], self.y[batch_indices]):
                # print(x,y)
                y_predicted = self.w * x + self.b
                # print("Predicted Y: ",y_predicted)
                err = self.error(y, y_predicted)
                dw = dw + self.grad_w(x,y, y_predicted)
                db = db + self.grad_b(y,y_predicted)
                total_error += err
            # print("dw: ",dw)
            self.w = self.w - self.learning_rate*dw/total_inputs
            self.b = self.b - self.learning_rate*db/total_inputs
            weights.append(self.w)
            biases.append(self.b)
            loss.append(total_error/total_inputs)
            # print(loss)
            # print(weights)
            # print(biases)
        print("Final W: ", self.w,
              " Final B: ", self.b)


if __name__ == "__main__":
    gd = batch_gradient_descent(100, 0, 5, 0.05)
    gd.gradient_descent()
