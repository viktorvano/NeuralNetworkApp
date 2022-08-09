package com.viktor.vano.neural.network.app.FFNN;

import com.sun.istack.internal.NotNull;

import java.awt.*;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.stream.IntStream;

import static com.viktor.vano.neural.network.app.FFNN.FileManagement.writeToFile;
import static com.viktor.vano.neural.network.app.FFNN.GeneralFunctions.showVectorValues;
import static com.viktor.vano.neural.network.app.FFNN.Weights.*;
import static com.viktor.vano.neural.network.app.Variables.isBusy;

public class NeuralNetwork {

    private TrainingThread trainingThread;
    public NeuralNetParameters neuralNetParameters;
    public ArrayList<Layer> m_layers; // m_layers[layerNum][neuronNum]

    private boolean netLoading;
    private boolean netTraining;
    private boolean stopTraining;

    private float m_loss;
    private float m_recentAverageLoss;

    public NeuralNetwork(@NotNull NeuralNetParameters neuralNetParameters)
    {
        this.neuralNetParameters = neuralNetParameters;
        this.netLoading = true;
        this.netTraining = false;
        this.stopTraining = false;
        this.m_loss = 0;
        this.m_recentAverageLoss = 0;
        int numLayers = neuralNetParameters.topology.size();
        System.out.println("Number of layers: " + numLayers);
        this.m_layers = new ArrayList<>();
        for (int layerNum = 0; layerNum < numLayers; layerNum++)
        {
            this.m_layers.add(new Layer());
            int numOutputs = layerNum == neuralNetParameters.topology.size() - 1 ? 0 : neuralNetParameters.topology.get(layerNum + 1);

            // We have made a new Layer, now fill it with neurons, and add a bias neuron to the layer.
            for (int neuronNum = 0; neuronNum <= neuralNetParameters.topology.get(layerNum); neuronNum++)
            {
                this.m_layers.get(this.m_layers.size()-1).add(new Neuron(this, numOutputs, neuronNum, layerNum));
                System.out.println("Made a neuron: " + neuronNum + " (in layer: " + layerNum +")");
            }

            // Force the bias node's output value to 1.0. It's last neuron created above
            m_layers.get(m_layers.size()-1).peekLast().setOutputValue(1.0f);
        }
        this.loadNeuronWeights();
        netLoading = false;
    }

    public boolean isNetLoading() {
        return netLoading;
    }
    public boolean isNetTraining(){
        return netTraining;
    }
    public void stopTraining()
    {
        stopTraining = true;
    }

    public void trainNeuralNetwork()
    {
        this.netTraining = true;
        this.trainingThread = new TrainingThread(this);
        this.trainingThread.start();
    }

    public void feedForward(ArrayList<Float> inputValues)
    {
        assert(inputValues.size() == m_layers.get(0).size() - 1);

        // Assign (latch) the input values into the input neurons
        IntStream.range(0, inputValues.size()).parallel().
                forEach(i -> m_layers.get(0).get(i).setOutputValue(inputValues.get(i)));

        // Forward propagate
        for (int layerNum = 1; layerNum < m_layers.size(); layerNum++)
        {
            Layer prevLayer = m_layers.get(layerNum - 1);

            final int finalLayerNum = layerNum;
            IntStream.range(0, m_layers.get(layerNum).size() - 1).parallel().
                    forEach(n -> m_layers.get(finalLayerNum).get(n).feedForward(prevLayer));
        }
    }

    public void backProp(ArrayList<Float> targetValues)
    {
        // Calculate overall net loss (RMS of output neuron losses)
        Layer outputLayer = m_layers.get(m_layers.size()-1);
        m_loss = 0.0f;

        IntStream.range(0, outputLayer.size() - 1).parallel().forEach(n ->
        {
            float delta = targetValues.get(n) - outputLayer.get(n).getOutputValue();
            m_loss += delta * delta;
        });
        m_loss /= outputLayer.size() - 1; //get average loss squared
        m_loss = (float)Math.sqrt(m_loss); // RMS

        // Implement a recent average measurement;

        m_recentAverageLoss = m_loss;

        // Calculate output layer gradients
        IntStream.range(0, outputLayer.size() - 1).parallel().forEach(n ->
        {
            outputLayer.get(n).calcOutputGradients(targetValues.get(n));
        });

        // Calculate gradients on hidden layers
        for (int layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
        {
            Layer hiddenLayer = m_layers.get(layerNum);
            Layer nextLayer = m_layers.get(layerNum + 1);

            hiddenLayer.parallelStream().forEach(neuron -> neuron.calcHiddenGradients(nextLayer));
        }

        // For all layers from outputs to first hidden layer.
        // update connection weights

        for (int layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
        {
            Layer layer = m_layers.get(layerNum);
            Layer prevLayer = m_layers.get(layerNum - 1);

            IntStream.range(0, layer.size() - 1).parallel().forEach(n -> layer.get(n).updateInputWeights(prevLayer));
        }
    }

    public void dream(@NotNull ArrayList<Float> targetDreamOutputs) throws Exception
    {
        if(targetDreamOutputs.size() != m_layers.get(m_layers.size() - 1).size() - 1)
            throw new Exception();

        //set output neuron values
        Layer outLayer = m_layers.get(m_layers.size() - 1);//last layer
        for (int n = 0; n < outLayer.size() - 1; n++)//last neuron is always const 1
        {
            outLayer.get(n).setOutputValue(targetDreamOutputs.get(n));
        }

        //set neuron outputs to zero except the output layer
        for (int layerNum = 0; layerNum <  m_layers.size() - 1; layerNum++)
        {
            Layer layer = m_layers.get(layerNum);
            for (int n = 0; n < layer.size() - 1; n++)//last neuron is always const 1
            {
                layer.get(n).setOutputValue(0f);
            }
        }

        //calc unbiased sums
        for (int layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
        {
            float unbiasedSumSum = 0f;
            Layer layer = m_layers.get(layerNum);//to calc unbiased sums
            Layer prevLayer = m_layers.get(layerNum - 1);//to get connection weights
            for (int n = 0; n < layer.size() - 1; n++)//last neuron is a bias neuron
            {
                layer.get(n).calcUnbiasedSum();
                unbiasedSumSum += layer.get(n).getUnbiasedSum();
            }
            //iterate in a loop dream process to set prev layers outputs
            ArrayList<ArrayList<Float>> weightFraction = new ArrayList<>();//<neurons in layer<connection weights>>
            for (int n = 0; n < prevLayer.size() - 1; n++)//last neuron is a bias neuron
            {
                weightFraction.add(new ArrayList<>());
                for(int c = 0; c < prevLayer.get(n).getOutputWeights().size(); c++)
                {
                    weightFraction.get(n).add(prevLayer.get(n).getOutputWeights().get(c).weight/1024f);
                }
            }
            float controlSum = 0f;
            //ArrayList<Float> lastFractionSum = new ArrayList<>();
            while (Math.abs(controlSum) < Math.abs(unbiasedSumSum))
            {
                //lastFractionSum.clear();
                float coefficient = Math.abs(controlSum - unbiasedSumSum);
                if(coefficient > 1f)
                {
                    coefficient = 1f;
                }
                if(coefficient < 0.0001)
                    break;

                for(int n = 0; n < prevLayer.size() - 1; n++)//increment weights
                {
                    float fractionSum = 0f;
                    for(int c = 0; c < weightFraction.get(n).size(); c++)
                    {
                        fractionSum += weightFraction.get(n).get(c) * layer.get(c).getUnbiasedSum() * coefficient;
                    }
                    //lastFractionSum.add(fractionSum);
                    prevLayer.get(n).setOutputValue(prevLayer.get(n).getOutputValue() + fractionSum);
                }

                //check sums of the next layer
                controlSum = 0f;
                for (int n = 0; n < layer.size() - 1; n++)
                {
                    controlSum += layer.get(n).feedUnbiasedSum(prevLayer);
                }
            }


            /*for(int n = 0; n < prevLayer.size() - 1; n++)//normalize results
            {
                prevLayer.get(n).setOutputValue((float)Math.tanh(prevLayer.get(n).getOutputValue()));
            }*/
            /*if(lastFractionSum.size() != 0)//should not happen
            {
                for(int n = 0; n < prevLayer.size() - 1; n++)//undo last increment of weights
                {
                    prevLayer.get(n).setOutputValue(prevLayer.get(n).getOutputValue() - lastFractionSum.get(n));
                }
            }*/
        }
    }

    public void getResults(ArrayList<Float> resultValues)
    {
        resultValues.clear();

        for (int n = 0; n < m_layers.get(m_layers.size()-1).size() - 1; n++)
        {
            resultValues.add(m_layers.get(m_layers.size()-1).get(n).getOutputValue());
        }
    }
    public float getNeuronOutput(int x, int y)
    {
        return m_layers.get(x).get(y).getOutputValue();
    }
    public float getRecentAverageLoss() { return m_recentAverageLoss; }

    public void saveNeuronWeights()
    {
        System.out.println("Saving Neuron Weights...");
        neuralNetParameters.neuronIndex = 0;
        allocateNewWeights(neuralNetParameters.topology, neuralNetParameters.weights);
        // Forward propagate
        for (int layerNum = 1; layerNum < m_layers.size(); layerNum++)
        {
            Layer prevLayer = m_layers.get(layerNum - 1);
            for (int n = 0; n < m_layers.get(layerNum).size() - 1; n++)
            {
                m_layers.get(layerNum).get(n).saveInputWeights(prevLayer);
            }
        }
        System.out.println("All Weights are Saved.");
        ZonedDateTime now = ZonedDateTime.now();
        writeToFile(neuralNetParameters.trainingStatusFilePath, now +
                "\nAverage Loss: " + String.format("%s", neuralNetParameters.averageLoss));
        Toolkit.getDefaultToolkit().beep();
    }

    public void loadNeuronWeights()
    {
        System.out.println("Loading Weights...");
        this.neuralNetParameters.neuronIndex = 0;
        allocateNewWeights(neuralNetParameters.topology, neuralNetParameters.weights);

        System.out.println("Reading file \"" + this.neuralNetParameters.weightsFilePath + "\"...");
        try
        {
            FileInputStream fi = new FileInputStream(this.neuralNetParameters.weightsFilePath);
            ObjectInputStream oi = new ObjectInputStream(fi);
            Object object;
            while(true)
            {
                try{
                    object = oi.readObject();
                }
                catch(IOException e){
                    break;
                }
                if(object != null)
                    this.neuralNetParameters.weights.set(this.neuralNetParameters.neuronIndex++,(Float) object);
            }

            oi.close();
            fi.close();
        }catch (Exception e)
        {
            System.out.println("Failed to read the \"" + this.neuralNetParameters.weightsFilePath + "\" file.");
        }

        if(this.neuralNetParameters.neuronIndex == this.neuralNetParameters.weights.size())
        {
            System.out.println("Weights loaded.");
        }else
        {
            System.out.println("Weights size did not match with the topology.");
            System.out.println("Loaded only what was there.");
            System.out.println("The rest are random weights.");
        }

        this.neuralNetParameters.neuronIndex = 0;
        // Forward propagate
        for (int layerNum = 1; layerNum < m_layers.size(); layerNum++)
        {
            System.out.println("Loading Layer: " + layerNum);
            Layer prevLayer = m_layers.get(layerNum - 1);
            for (int n = 0; n < m_layers.get(layerNum).size() - 1; n++)
            {
                m_layers.get(layerNum).get(n).loadInputWeights(prevLayer);
            }
        }
        Toolkit.getDefaultToolkit().beep();
    }

    private class TrainingThread extends Thread
    {
        NeuralNetwork myNet;
        NeuralNetParameters netObjects;

        private TrainingThread(NeuralNetwork net){
            this.myNet = net;
            this.netObjects = net.neuralNetParameters;
        }

        @Override
        public void run() {
            super.run();
            trainNeuralNet();
        }

        private void trainNeuralNet()
        {
            netObjects.input.clear();
            netObjects.target.clear();
            netObjects.result.clear();
            neuralNetParameters.trainingPass = 0;

            loadTrainingDataFromFile(neuralNetParameters);

            System.out.println("Training started\n");
            this.netObjects.averageLoss = 1.0f;
            float currentTrainingLoss;
            float quickSaveLossValue = 0.5f;
            boolean repeatTrainingCycle = false;
            while (true)
            {
                netObjects.trainingPass++;
                System.out.println("Pass: " + netObjects.trainingPass);

                isBusy = true;
                //Get new input data and feed it forward:
                if(!repeatTrainingCycle)
                    netObjects.trainData.getNextInputs(netObjects);
                showVectorValues("Inputs:", netObjects.input);
                myNet.feedForward(netObjects.input);

                // Train the net what the outputs should have been:
                if(!repeatTrainingCycle)
                    netObjects.trainData.getTargetOutputs(netObjects);
                showVectorValues("Targets: ", netObjects.target);
                assert(netObjects.target.size() == netObjects.topology.get(netObjects.topology.size()-1));
                myNet.backProp(netObjects.target);//This function alters neurons

                // Collect the net's actual results:
                myNet.getResults(netObjects.result);
                isBusy = false;
                showVectorValues("Outputs: ", netObjects.result);


                // Report how well the training is working, averaged over recent samples:
                System.out.println("Net recent average loss: " + myNet.getRecentAverageLoss() + "\n\n");

                currentTrainingLoss = myNet.getRecentAverageLoss();
                this.netObjects.averageLoss = 0.99f*this.netObjects.averageLoss + 0.01f*currentTrainingLoss;
                if(this.netObjects.averageLoss < this.netObjects.velocity)
                    this.netObjects.velocity = this.netObjects.averageLoss;
                if(this.netObjects.averageLoss < this.netObjects.momentum)
                    this.netObjects.momentum = this.netObjects.averageLoss;
                System.out.println("Net average loss: " + this.netObjects.averageLoss + "\n\n");
                repeatTrainingCycle = currentTrainingLoss > this.netObjects.averageLoss;

                if(this.netObjects.averageLoss < netObjects.trainingExitLoss
                  && netObjects.trainingPass > netObjects.minTrainingPasses)
                {
                    System.out.println("Exit due to low loss :D\n\n");
                    myNet.saveNeuronWeights();
                    break;
                }if(this.netObjects.averageLoss < quickSaveLossValue)
                {
                    quickSaveLossValue = this.netObjects.averageLoss /2f;
                    myNet.saveNeuronWeights();
                }if(netObjects.trainingPass > netObjects.maxTrainingPasses)
                {
                    System.out.println("Training passes were exceeded...\n\n");
                    myNet.saveNeuronWeights();
                    break;
                }

                if(stopTraining)
                    break;
            }
            System.out.println("Training done.\n");
            stopTraining = false;
            this.myNet.netTraining = false;
            System.out.println("Neural Network loaded.");
        }

    }
}
