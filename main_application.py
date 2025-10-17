"""
main.py
Main application for ESP32 Neural Network Inference
Upload this file to your ESP32 using Thonny
"""

import time
import math
from neural_network import NeuralNetwork


def generate_sine_wave(frequency, num_samples=32, noise=0.0):
    """
    Generate a sine wave signal
    
    Args:
        frequency: Frequency in Hz
        num_samples: Number of samples to generate
        noise: Amount of random noise to add (0-1)
    
    Returns:
        List of signal values
    """
    signal = []
    for i in range(num_samples):
        x = (i / num_samples) * 2 * math.pi
        value = math.sin(frequency * x)
        
        # Add noise if requested
        if noise > 0:
            # Simple pseudo-random noise
            noise_val = ((time.ticks_us() % 1000) / 1000.0 - 0.5) * noise
            value += noise_val
        
        signal.append(value)
    
    return signal


def print_header():
    """Print application header"""
    print("\n" + "="*60)
    print("  ESP32 NEURAL NETWORK INFERENCE")
    print("  Real-time Classification System")
    print("="*60)


def print_prediction(frequency, predicted_class, probabilities, inference_time):
    """
    Print formatted prediction results
    
    Args:
        frequency: Input frequency
        predicted_class: Predicted class (0 or 1)
        probabilities: List of class probabilities
        inference_time: Time taken for inference in ms
    """
    # Determine expected class
    expected_class = 0 if frequency < 3.5 else 1
    is_correct = (predicted_class == expected_class)
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    
    print("\n" + "-"*60)
    print(f"Input: {frequency:.1f} Hz sine wave")
    print(f"Inference time: {inference_time} ms")
    print(f"\nClass probabilities:")
    print(f"  Class 0 (Low freq):  {probabilities[0]*100:5.2f}%")
    print(f"  Class 1 (High freq): {probabilities[1]*100:5.2f}%")
    print(f"\nPredicted: Class {predicted_class}")
    print(f"Expected:  Class {expected_class}")
    print(f"Result: {status}")
    print("-"*60)


def run_test_suite(nn):
    """
    Run a comprehensive test suite
    
    Args:
        nn: Neural network instance
    """
    print("\n" + "="*60)
    print("RUNNING TEST SUITE")
    print("="*60)
    
    test_frequencies = [
        (1.5, "Very Low"),
        (2.0, "Low"),
        (2.5, "Medium-Low"),
        (3.0, "Boundary-Low"),
        (3.5, "Boundary"),
        (4.0, "Boundary-High"),
        (4.5, "Medium-High"),
        (5.0, "High"),
        (5.5, "Very High")
    ]
    
    correct_predictions = 0
    total_predictions = len(test_frequencies)
    total_time = 0
    
    for freq, description in test_frequencies:
        # Generate test signal
        signal = generate_sine_wave(freq)
        
        # Run inference
        start_time = time.ticks_ms()
        probabilities = nn.predict(signal)
        inference_time = time.ticks_diff(time.ticks_ms(), start_time)
        
        predicted_class = 0 if probabilities[0] > probabilities[1] else 1
        expected_class = 0 if freq < 3.5 else 1
        
        is_correct = (predicted_class == expected_class)
        if is_correct:
            correct_predictions += 1
        
        total_time += inference_time
        
        # Print result
        status = "✓" if is_correct else "✗"
        print(f"{status} {freq} Hz ({description:13s}) -> Class {predicted_class} "
              f"[{probabilities[predicted_class]*100:5.1f}%] ({inference_time} ms)")
    
    # Print summary
    accuracy = (correct_predictions / total_predictions) * 100
    avg_time = total_time / total_predictions
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests:     {total_predictions}")
    print(f"Correct:         {correct_predictions}")
    print(f"Wrong:           {total_predictions - correct_predictions}")
    print(f"Accuracy:        {accuracy:.1f}%")
    print(f"Avg time:        {avg_time:.1f} ms")
    print(f"Min time:        ~{avg_time:.0f} ms")
    print(f"Max time:        ~{avg_time:.0f} ms")
    print("="*60)


def interactive_mode(nn):
    """
    Interactive mode - continuously classify random frequencies
    
    Args:
        nn: Neural network instance
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Press Ctrl+C to stop")
    print("\nClassifying random frequencies every 3 seconds...\n")
    
    test_count = 0
    correct_count = 0
    
    try:
        while True:
            test_count += 1
            
            # Generate random frequency between 1.0 and 6.0 Hz
            # Using time as pseudo-random seed
            random_val = (time.ticks_us() % 500) / 100.0
            frequency = 1.0 + random_val
            
            # Generate signal with small noise
            signal = generate_sine_wave(frequency, noise=0.05)
            
            # Run inference
            start_time = time.ticks_ms()
            probabilities = nn.predict(signal)
            inference_time = time.ticks_diff(time.ticks_ms(), start_time)
            
            predicted_class = 0 if probabilities[0] > probabilities[1] else 1
            expected_class = 0 if frequency < 3.5 else 1
            
            if predicted_class == expected_class:
                correct_count += 1
            
            # Print compact result
            status = "✓" if predicted_class == expected_class else "✗"
            accuracy = (correct_count / test_count) * 100
            
            print(f"[Test {test_count:3d}] {frequency:4.1f} Hz -> "
                  f"Class {predicted_class} {status} | "
                  f"Confidence: {probabilities[predicted_class]*100:5.1f}% | "
                  f"Time: {inference_time:2d} ms | "
                  f"Accuracy: {accuracy:5.1f}%")
            
            # Wait before next test
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("INTERACTIVE MODE STOPPED")
        print("="*60)
        print(f"Total tests: {test_count}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {(correct_count/test_count)*100:.1f}%")
        print("="*60)


def main():
    """Main function"""
    
    # Print header
    print_header()
    
    # Initialize neural network
    print("\nInitializing neural network...")
    nn = NeuralNetwork()
    
    # Load pre-trained weights
    if not nn.load_weights('model_weights.json'):
        print("\n✗ ERROR: Could not load model weights!")
        print("Make sure 'model_weights.json' is uploaded to ESP32")
        return
    
    print("\n✓ Neural network ready!")
    
    # Run test suite
    print("\nRunning initial tests...")
    time.sleep(1)
    run_test_suite(nn)
    
    # Wait a moment
    time.sleep(2)
    
    # Start interactive mode
    interactive_mode(nn)


# Run main application
if __name__ == "__main__":
    main()
