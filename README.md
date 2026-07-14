This project presents a Neuromorphic Vision 
Processor built using Spiking Neural Networks 
(SNN) that mimics the biological functioning 
of the human brain to recognize handwritten 
digits with high accuracy and ultra-low 
energy consumption.

Unlike traditional Convolutional Neural 
Networks (CNN) that process information 
continuously using dense computations, our 
system uses Leaky Integrate-and-Fire (LIF) 
neurons that communicate through discrete 
electrical pulses called spikes, exactly 
like real neurons in the human brain. This 
event-driven approach makes the system 
significantly more energy efficient.

The core model was implemented using snnTorch 
and PyTorch frameworks with a two-layer 
architecture consisting of 784 input neurons, 
1000 LIF hidden neurons, and 10 output LIF 
neurons representing digits 0 through 9. 
The network processes each image over 25 
simulation timesteps with a membrane decay 
rate of 0.95, using surrogate gradient 
descent with Adam optimizer for training.

The model was trained on the MNIST dataset 
containing 60,000 handwritten digit images 
across 10 epochs, achieving 97.8% test 
accuracy which significantly exceeds the 
targeted 92% benchmark.

A major finding is that our SNN consumes 
only 135 nanojoules per inference compared 
to 18,604 nanojoules for an equivalent CNN, 
making our system 137 times more energy 
efficient through sparse spike-based 
computation.

A professional Flask web application was 
developed featuring an interactive digit 
drawing canvas, image upload, real-time 
webcam detection, spike visualization, 
membrane voltage graphs, and energy 
comparison metrics.

This project demonstrates that neuromorphic 
computing is ideal for next-generation 
low-power applications including mobile 
devices, autonomous drones, medical implants, 
and edge computing systems, bringing us 
closer to truly brain-inspired artificial 
intelligence.
