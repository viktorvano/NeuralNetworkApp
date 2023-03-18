package com.viktor.vano.neural.network.app.FFNN;

import com.sun.istack.internal.NotNull;

import java.awt.*;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.IntStream;

import static com.viktor.vano.neural.network.app.FFNN.FileManagement.writeToFile;
import static com.viktor.vano.neural.network.app.FFNN.GeneralFunctions.showVectorValues;
import static com.viktor.vano.neural.network.app.FFNN.Weights.*;
import static com.viktor.vano.neural.network.app.Variables.update;
import static com.viktor.vano.neural.network.app.Variables.updateInputSliders;

public class NeuralNetwork {

    private TrainingThread trainingThread;
    public NeuralNetParameters neuralNetParameters;
    public ArrayList<Layer> m_layers; // m_layers[layerNum][neuronNum]

    private boolean netLoading;
    private boolean netTraining;
    private boolean stopTraining;

    private float m_loss;
    private float m_recentAverageLoss;
    private float imaginationProgress;
    private boolean isImaginationRunning;
    private float trainingProgress;

    public NeuralNetwork(@NotNull NeuralNetParameters neuralNetParameters)
    {
        this.neuralNetParameters = neuralNetParameters;
        this.netLoading = true;
        this.netTraining = false;
        this.stopTraining = false;
        this.m_loss = 0;
        this.m_recentAverageLoss = 0;
        this.imaginationProgress = 0.0f;
        this.trainingProgress = 0.0f;
        this.isImaginationRunning = false;
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

    public float getImaginationProgress()
    {
        return this.imaginationProgress;
    }

    public float getTrainingProgress()
    {
        return this.trainingProgress;
    }

    public boolean isImaginationRunning()
    {
        return this.isImaginationRunning;
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
        this.trainingThread.setName("Training Thread " + Math.round(Math.random()*1000.0));
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

    public ArrayList<Float> imagine(@NotNull final ArrayList<Float> targetImaginationOutputs, float minInValue, float maxInValue) throws Exception
    {
        if(targetImaginationOutputs.size() != m_layers.get(m_layers.size() - 1).size() - 1)
            throw new Exception();

        this.isImaginationRunning = true;
        final int populationSize = 500;
        final int survivorCount = 50;
        final float exitLoss = 0.01f;
        int generation = 0;
        final int maxGenerations = 3000;
        this.imaginationProgress = 0.0f;
        ArrayList<Individual> individuals = new ArrayList<>();
        ArrayList<Individual> survivors = new ArrayList<>();

        for (int i=0; i<populationSize; i++)
        {
            individuals.add(new Individual(targetImaginationOutputs, this, minInValue, maxInValue));
        }
        sortTheBestIndividuals(individuals, survivors, survivorCount);
        plotSurvivorLosses(survivors, generation);

        for(int i=0; i<maxGenerations; i++)
        {
            generation++;
            System.out.println("Generation: " + generation);
            populateGeneration(individuals,
                                survivors,
                                populationSize,
                                this,
                                targetImaginationOutputs,
                                minInValue,
                                maxInValue);
            sortTheBestIndividuals(individuals, survivors, survivorCount);
            plotSurvivorLosses(survivors, generation);

            System.out.println("Lowest loss of generation " + generation + " : " + survivors.get(0).getLoss());
            update = true;
            updateInputSliders = true;
            this.imaginationProgress = exitLoss / survivors.get(0).getLoss();
            if(this.imaginationProgress >= 1.0f)
            {
                System.out.println("Imagination ended due to low loss with generation: " + generation);
                break;
            }
        }

        this.isImaginationRunning = false;
        return survivors.get(0).input;//the best of the best
    }

    private static void sortTheBestIndividuals(ArrayList<Individual> individuals, ArrayList<Individual> survivors, final float survivorCount)
    {
        survivors.clear();

        //sort the best individuals
        float[] losses = new float[individuals.size()];
        for (int i=0; i<individuals.size(); i++)
        {
            losses[i] = individuals.get(i).getLoss();
        }
        Arrays.sort(losses);

        //find the best individuals based on the lowest loss
        for (int i=0; i<survivorCount; i++)
        {
            for(Individual individual : individuals)
            {
                if(individual.getLoss() == losses[i])
                {
                    survivors.add(individual);
                    break;
                }
            }
        }
    }

    private static void plotSurvivorLosses(final ArrayList<Individual> survivors, final int generation)
    {
        int index = 0;
        for(Individual survivor : survivors)
        {
            System.out.println("Survivor [" + index + "] of generation [" + generation + "]: " + survivor.getLoss());
            index++;
        }
    }

    private void populateGeneration(ArrayList<Individual> individuals,
                                           ArrayList<Individual> survivors,
                                           final int populationSize,
                                           NeuralNetwork neuralNetwork,
                                           final ArrayList<Float> target,
                                           float minInValue,
                                           float maxInValue)
    {
        individuals.clear();
        //add the survivors to the new generation
        for(Individual survivor : survivors)
        {
            individuals.add(survivor);
        }

        //fill the population with new variants of the individuals
        do {
            int randValue =  (int)(Math.round(Math.random()*50.0));
            if(randValue < 10)
            {
                individuals.add(new Individual(target, neuralNetwork, minInValue, maxInValue));
            }else if(randValue < 20)
            {
                //alter one input value - Random iteration
                int randomSurvivorIndex = (int)(Math.round(Math.random()*survivors.size()));
                if(randomSurvivorIndex == survivors.size())
                    randomSurvivorIndex--;
                Individual individual = new Individual(target, neuralNetwork, minInValue, maxInValue);
                final int inputSize = individual.input.size();
                for(int i=0; i<inputSize; i++)//copy all genes
                {
                    individual.input.set(i, individuals.get(randomSurvivorIndex).input.get(i));
                }

                int randomGeneIndex = (int)(Math.round(Math.random()*individual.input.size()));
                if(randomGeneIndex == individual.input.size())
                    randomGeneIndex--;
                float geneValue = individual.input.get(randomGeneIndex);
                final float delta = 0.001f;
                if(geneValue + delta >= maxInValue)
                {
                    geneValue -= delta;
                } else if (geneValue - delta <= minInValue) {
                    geneValue += delta;
                } else
                {
                    if(Math.random() > 0.5)
                    {
                        geneValue += delta;
                    }else
                    {
                        geneValue -= delta;
                    }
                }
                individual.input.set(randomGeneIndex, geneValue);
                //add the individual
                individuals.add(individual);
            }else if(randValue < 30)
            {
                //randomly change one input value - Mutation
                int randomSurvivorIndex = (int)(Math.round(Math.random()*survivors.size()));
                if(randomSurvivorIndex == survivors.size())
                    randomSurvivorIndex--;
                Individual individual = new Individual(target, neuralNetwork, minInValue, maxInValue);
                final int inputSize = individual.input.size();
                for(int i=0; i<inputSize; i++)//copy all genes
                {
                    individual.input.set(i, individuals.get(randomSurvivorIndex).input.get(i));
                }

                //mutate a random gene
                int randomGeneIndex = (int)(Math.round(Math.random()*individual.input.size()));
                if(randomGeneIndex == individual.input.size())
                    randomGeneIndex--;
                individual.input.set(randomGeneIndex, individual.randomValue(minInValue, maxInValue));
                individuals.add(individual);
            }else if(randValue < 40)
            {
                //Uniform Crossover
                int randomSurvivorIndexA = (int)(Math.round(Math.random()*survivors.size()));
                if(randomSurvivorIndexA == survivors.size())
                    randomSurvivorIndexA--;

                int randomSurvivorIndexB = (int)(Math.round(Math.random()*survivors.size()));
                if(randomSurvivorIndexB == survivors.size())
                    randomSurvivorIndexB--;
                final Individual individualA = survivors.get(randomSurvivorIndexA);
                final Individual individualB = survivors.get(randomSurvivorIndexB);
                Individual individual = new Individual(target, neuralNetwork, minInValue, maxInValue);
                final int inputSize = individual.input.size();
                for(int i=0; i<inputSize; i++)
                {
                    //set each input one by one from two random survivors
                    if(Math.random() < 0.5)
                        individual.input.set(i, individualA.input.get(i));
                    else
                        individual.input.set(i, individualB.input.get(i));
                }
                individuals.add(individual);
            }else
            {
                //Orgy
                Individual individual = new Individual(target, neuralNetwork, minInValue, maxInValue);
                final int inputSize = individual.input.size();
                for(int i=0; i<inputSize; i++)
                {
                    //set each input one by one from a random survivor
                    int randomSurvivorIndex = (int)(Math.round(Math.random()*survivors.size()));
                    if(randomSurvivorIndex == survivors.size())
                        randomSurvivorIndex--;
                    Individual randomSurvivor = survivors.get(randomSurvivorIndex);
                    individual.input.set(i, randomSurvivor.input.get(i));
                }
                individuals.add(individual);
            }
        }while (individuals.size() != populationSize);
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
            trainingProgress = 0.0f;
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
            trainingProgress = 0.0f;

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

                update = false;
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
                update = true;
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

                trainingProgress = netObjects.trainingExitLoss / this.netObjects.averageLoss;

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

    private class Individual
    {
        private ArrayList<Float> input, result;
        private final ArrayList<Float> target;
        private NeuralNetwork neuralNetwork;
        private float loss, min, max;

        public Individual(ArrayList<Float> target, @NotNull NeuralNetwork neuralNetwork, float min, float max)
        {
            this.target = target;
            this.input = new ArrayList<>();
            this.result = new ArrayList<>();
            this.neuralNetwork = neuralNetwork;
            this.min = min;
            this.max = max;

            for(int i=0; i<neuralNetwork.neuralNetParameters.topology.get(0); i++)
            {
                input.add(randomValue(this.min, this.max));
            }
        }

        private void calcLoss()
        {
            this.neuralNetwork.feedForward(input);
            this.neuralNetwork.getResults(result);
            if(result.size() == 0 || target.size() == 0 ||
              (target.size() != result.size()))
            {
                this.loss = -1.0f;
                return;
            }

            float delta = 0.0f;
            for(int i=0; i<target.size(); i++)
            {
                delta += Math.abs(result.get(i) - target.get(i));
            }

            this.loss = delta;
        }

        public float getLoss()
        {
            calcLoss();
            return this.loss;
        }

        public ArrayList<Float> getResult()
        {
            return result;
        }

        public ArrayList<Float> getTarget()
        {
            return target;
        }

        public float randomValue(float min, float max)
        {
            float range = max - min;
            float randomValue = ((float)Math.random() * range) - min;
            return randomValue;
        }
    }
}
