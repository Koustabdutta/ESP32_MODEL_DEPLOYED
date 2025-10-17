"""
neural_network.py
Neural Network Library for MicroPython/ESP32
Upload this file to your ESP32 using Thonny
"""

import math
import json


class NeuralNetwork:
    """
    Simple feedforward neural network implementation
    Optimized for microcontrollers
    """
    
    def __init__(self):
        """Initialize neural network"""
        self.layers = []
        self.layer_info = []
        print("Neural Network initialized")
    
    def load_weights(self, filename='model_weights.json'):
        """
        Load pre-trained weights from JSON file
        
        Args:
            filename: Path to weights file on ESP32 filesystem
        """
        try:
            print(f"Loading weights from {filename}...")
            
            with open(filename, 'r') as f:
                weights_dict = json.load(f)
            
            # Sort layers by name to ensure correct order
            layer_names = sorted(weights_dict.keys())
            
            for layer_name in layer_names:
                layer_data = weights_dict[layer_name]
                
                layer = {
                    'name': layer_name,
                    'weights': layer_data['weights'],
                    'bias': layer_data['bias'],
                    'input_size': layer_data['input_size'],
                    'output_size': layer_data['output_size']
                }
                
                self.layers.append(layer)
                self.layer_info.append(
                    f"{layer_name}: {layer['input_size']} -> {layer['output_size']}"
                )
            
            print(f"✓ Loaded {len(self.layers)} layers")
            for info in self.layer_info:
                print(f"  {info}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading weights: {e}")
            return False
    
    def relu(self, x):
        """ReLU activation function"""
        return max(0.0, x)
    
    def softmax(self, x):
        """
        Softmax activation function
        Converts raw scores to probabilities
        """
        # Subtract max for numerical stability
        max_x = max(x)
        exp_x = [math.exp(val - max_x) for val in x]
        sum_exp = sum(exp_x)
        return [val / sum_exp for val in exp_x]
    
    def dense_layer(self, inputs, weights, bias, activation=None):
        """
        Compute output of a dense (fully connected) layer
        
        Args:
            inputs: Input values
            weights: Weight matrix [input_size][output_size]
            bias: Bias vector [output_size]
            activation: Activation function ('relu' or None)
        
        Returns:
            Output values
        """
        outputs = []
        
        # For each output neuron
        for neuron_idx in range(len(weights[0])):
            # Compute weighted sum
            weighted_sum = bias[neuron_idx]
            
            for input_idx, input_val in enumerate(inputs):
                weighted_sum += input_val * weights[input_idx][neuron_idx]
            
            # Apply activation function
            if activation == 'relu':
                weighted_sum = self.relu(weighted_sum)
            
            outputs.append(weighted_sum)
        
        return outputs
    
    def predict(self, input_data):
        """
        Run forward pass through the network
        
        Args:
            input_data: Input array (must match network input size)
        
        Returns:
            Output probabilities for each class
        """
        if not self.layers:
            raise ValueError("No layers loaded. Call load_weights() first.")
        
        # Convert input to list if needed
        x = list(input_data)
        
        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            is_last_layer = (i == len(self.layers) - 1)
            
            # Use ReLU for hidden layers, no activation for output layer
            activation = None if is_last_layer else 'relu'
            
            x = self.dense_layer(
                x,
                layer['weights'],
                layer['bias'],
                activation
            )
        
        # Apply softmax to output layer
        x = self.softmax(x)
        
        return x
    
    def predict_class(self, input_data):
        """
        Predict class label and confidence
        
        Args:
            input_data: Input array
        
        Returns:
            (predicted_class, confidence) tuple
        """
        probabilities = self.predict(input_data)
        predicted_class = probabilities.index(max(probabilities))
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence
    
    def get_info(self):
        """Get network architecture information"""
        info = "Neural Network Architecture:\n"
        for layer_info in self.layer_info:
            info += f"  {layer_info}\n"
        return info


def test_network():
    """Test function to verify neural network works"""
    print("\n" + "="*50)
    print("Testing Neural Network")
    print("="*50)
    
    # Create and load network
    nn = NeuralNetwork()
    
    if not nn.load_weights('model_weights.json'):
        print("Failed to load weights!")
        return
    
    print("\n" + nn.get_info())
    
    # Test with random data
    print("\nTesting with random input...")
    test_input = [math.sin(i * 0.1) for i in range(32)]
    
    try:
        output = nn.predict(test_input)
        predicted_class, confidence = nn.predict_class(test_input)
        
        print(f"Output probabilities: {[f'{p:.4f}' for p in output]}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("\n✓ Neural network is working correctly!")
        
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
    
    print("="*50)


# If running this file directly, run test
if __name__ == "__main__":
    test_network()
