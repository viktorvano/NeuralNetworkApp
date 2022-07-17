package com.viktor.vano.neural.network.app.FFNN;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class Neuron {
    private NeuralNetwork neuralNetwork;

    private float m_outputValue;
    private ArrayList<Connection> m_outputWeights;
    private int m_myIndex;
    private float m_gradient;

    private int m_layerNumber;

    private float unbiasedSum;

    public Neuron(NeuralNetwork neuralNetwork, int numOutputs, int myIndex, int layerNumber)
    {
        this.neuralNetwork = neuralNetwork;
        this.m_layerNumber = layerNumber;
        m_outputWeights = new ArrayList<>();
        m_outputWeights.clear();

        for (int c = 0; c < numOutputs; c++)
        {
            m_outputWeights.add(new Connection());
        }

        m_myIndex = myIndex;
    }

    public void calcUnbiasedSum()
    {
        int previousLayerIndex = m_layerNumber - 1;
        int prevLayerLastNeuronIndex = neuralNetwork.m_layers.get(previousLayerIndex).size() - 1;
        unbiasedSum = 0.5f * (float)Math.log((1.0+(double)m_outputValue)/(1.0-(double)m_outputValue))//tanh^-1(x)
                        - neuralNetwork.m_layers.get(previousLayerIndex).get(prevLayerLastNeuronIndex)
                            .m_outputWeights.get(m_myIndex).weight;
    }

    public float getUnbiasedSum()
    {
        return unbiasedSum;
    }

    public ArrayList<Connection> getOutputWeights()
    {
        return m_outputWeights;
    }

    public float feedUnbiasedSum(Layer prevLayer)
    {
        float unbiasedSum = 0.0f;

        for (int n = 0; n < prevLayer.size() - 1; n++)//last neuron is a bias neuron
        {
            unbiasedSum += prevLayer.get(n).getOutputValue() * prevLayer.get(n).m_outputWeights.get(m_myIndex).weight;
        }

        return unbiasedSum;
    }

    public void setOutputValue(float value) { m_outputValue = value; }
    public float getOutputValue() { return m_outputValue; }
    public void feedForward(Layer prevLayer)
    {
        float sum = 0.0f;

        // Sum the previous layer's outputs (which are inputs)
        // Include the bias node from the previous layer.

        for (int n = 0; n < prevLayer.size(); n++)
        {
            sum += prevLayer.get(n).getOutputValue() * prevLayer.get(n).m_outputWeights.get(m_myIndex).weight;
        }

        m_outputValue = Neuron.transferFunction(sum);
    }

    public void calcOutputGradients(float targetValue)
    {
        float delta = targetValue - m_outputValue;
        m_gradient = delta * transferFunctionDerivative(m_outputValue);
    }

    public void calcHiddenGradients(Layer nextLayer)
    {
        float dow = sumDOW(nextLayer);
        m_gradient = dow * transferFunctionDerivative(m_outputValue);
    }

    public void updateInputWeights(Layer prevLayer)
    {
        // The weights to updated are in the Connection container
        // in the neurons in the preceding layer
        for (int n = 0; n < prevLayer.size(); n++)
        {
            Neuron neuron = prevLayer.get(n);
            float oldDeltaWeight = neuron.m_outputWeights.get(m_myIndex).deltaWeight;

            float newDeltaWeight =
                    // Individual input, magnified by the gradient and train rate:
                    this.neuralNetwork.neuralNetParameters.velocity //ETA 0.0==slowlearner; 0.2==medium learner; 1.0==reckless learner
                            * neuron.getOutputValue()
                            * m_gradient
                            // Also add momentum = a fraction of the previous delta weight
                            + this.neuralNetwork.neuralNetParameters.momentum //ALPHA 0.0==no momentum; 0.5==moderate momentum
                            * oldDeltaWeight;
            neuron.m_outputWeights.get(m_myIndex).deltaWeight = newDeltaWeight;
            neuron.m_outputWeights.get(m_myIndex).weight += newDeltaWeight;
        }
    }

    public void saveInputWeights(Layer prevLayer)
    {
        // The weights to updated are in the Connection container
        // in the neurons in the preceding layer

        for (int n = 0; n < prevLayer.size(); n++)
        {
            Neuron neuron = prevLayer.get(n);
            neuralNetwork.neuralNetParameters.weights.set(neuralNetwork.neuralNetParameters.neuronIndex, neuron.m_outputWeights.get(m_myIndex).weight);
            neuralNetwork.neuralNetParameters.neuronIndex++;
        }

        if (neuralNetwork.neuralNetParameters.neuronIndex == neuralNetwork.neuralNetParameters.weights.size())
        {
            //save weights from Weights[] to a file
            //save weights from Weights[] to a file
            System.out.println("Saving weights to weights.dat...");
            try
            {
                File file = new File(neuralNetwork.neuralNetParameters.weightsFilePath);
                file.createNewFile();
                FileOutputStream f = new FileOutputStream(file);
                ObjectOutputStream o = new ObjectOutputStream(f);
                for (Float weight : neuralNetwork.neuralNetParameters.weights) o.writeObject(weight);
                o.close();
                f.close();
            }catch (Exception e)
            {
                System.out.println("Failed to create the \"" + neuralNetwork.neuralNetParameters.weightsFilePath + "\" file.");
            }
        }
    }

    public void loadInputWeights(Layer prevLayer)
    {
        for (Neuron neuron : prevLayer)
        {
            if(neuralNetwork.neuralNetParameters.weights.get(neuralNetwork.neuralNetParameters.neuronIndex) != null)
                neuron.m_outputWeights.get(m_myIndex).weight = neuralNetwork.neuralNetParameters.weights.get(neuralNetwork.neuralNetParameters.neuronIndex);
            this.neuralNetwork.neuralNetParameters.neuronIndex++;
        }
    }

    private float sumDOW(Layer nextLayer)
    {
        float sum = 0.0f;

        // Sum our contributions of the errors at the nodes we feed
        for (int n = 0; n < nextLayer.size() - 1; n++)
        {
            sum += m_outputWeights.get(n).weight * nextLayer.get(n).m_gradient;
        }

        return sum;
    }

    private static float transferFunction(float x)
    {
        // tanh - output range [-1.0..1.0]
        return (float)Math.tanh(x);
    }

    private float transferFunctionDerivative(float x)
    {
        // tanh derivative
        return (float) (1.0f - (float)Math.pow(Math.tanh(x), 2.0));// approximation return 1.0 - x*x;
    }
}
