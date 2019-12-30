# This piece of code evaluates the accuracy on test dataset.
print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

# This piece of code saves the model to disk.
print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))
