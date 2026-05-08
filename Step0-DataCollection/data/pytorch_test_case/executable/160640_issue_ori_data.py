from torch import simpletorch

neural_net = simpletorch.NeuralNet(
    inputs=100, 
    outputs=100, 
    layers=5)

training_data = simpletorch.image_data("./images")

neural_net.train(
    epochs=3, 
    training_data=training_data)

answer = neuralnet.guess("./test_images/test_image.jpg")