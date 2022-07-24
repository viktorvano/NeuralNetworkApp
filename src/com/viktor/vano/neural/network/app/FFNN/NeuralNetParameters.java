package com.viktor.vano.neural.network.app.FFNN;

import com.sun.istack.internal.NotNull;

import java.util.ArrayList;

import static com.viktor.vano.neural.network.app.FFNN.Weights.loadTopology;

public class NeuralNetParameters {
    public Integer trainingSamplesCount;
    public Integer inputNodes;
    public Integer outputNodes;
    public Float velocity; //ETA overall net learning rate [0.0..1.0]
    public Float momentum; //ALPHA momentum multiplier of last deltaWeight [0.0..n]

    public String weightsFilePath;
    public String trainingFilePath;
    public String topologyFilePath;
    public String trainingStatusFilePath;
    public ArrayList<Integer> topology;
    public ArrayList<ArrayList<Float>> learningInputs;
    public ArrayList<ArrayList<Float>> learningOutputs;
    public ArrayList<Float> weights;
    public Integer neuronIndex;
    public Integer trainingLine;
    public ArrayList<Float> input, target, result;
    public Integer trainingPass;

    public Float trainingExitLoss;
    public Float averageLoss;
    public Integer minTrainingPasses;
    public Integer maxTrainingPasses;
    public TrainingData trainData;

    public NeuralNetParameters(@NotNull String topologyFilePath, @NotNull String trainingFilePath, @NotNull String weightsFilePath, @NotNull String trainingStatusFilePath, float velocity, float momentum, float trainingExitLoss, int minTrainingPasses, int maxTrainingPasses)
    {
        this.trainingSamplesCount = 0;
        this.inputNodes = 0;
        this.outputNodes = 0;
        this.velocity = velocity; // overall net learning rate [0.0..1.0]
        this.momentum = momentum; // momentum multiplier of last deltaWeight [0.0..n]
        this.averageLoss = 1.0f;

        this.weightsFilePath = weightsFilePath;
        this.trainingFilePath = trainingFilePath;
        this.topologyFilePath = topologyFilePath;
        this.trainingStatusFilePath = trainingStatusFilePath;
        this.topology = new ArrayList<>();
        this.learningInputs = new ArrayList<>();
        this.learningOutputs = new ArrayList<>();
        this.weights = new ArrayList<>();
        this.neuronIndex = 0;
        this.trainingLine = 0;
        this.input = new ArrayList<>();;
        this.target = new ArrayList<>();;
        this.result = new ArrayList<>();
        this.trainingPass = 0;
        this.trainingExitLoss = trainingExitLoss;
        this.minTrainingPasses = minTrainingPasses;
        this.maxTrainingPasses = maxTrainingPasses;
        this.trainData = new TrainingData();


        loadTopology(this);
        if (topology.size() < 3)
        {
            System.out.println("Topology ERROR:\nTopology is too short, may miss some layer.");
            System.exit(-236);
        }
    }
}
