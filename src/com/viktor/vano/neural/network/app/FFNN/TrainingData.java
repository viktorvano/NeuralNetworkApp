package com.viktor.vano.neural.network.app.FFNN;

import java.util.ArrayList;

public class TrainingData {
    public ArrayList<String> InputLabels, OutputLabels;

    public TrainingData()
    {
        InputLabels = new ArrayList<>();
        OutputLabels = new ArrayList<>();
    }
    public static int getNextInputs(NeuralNetParameters neuralNetParameters)
    {
        neuralNetParameters.input.clear();

        neuralNetParameters.trainingLine = (int)Math.round(Math.random()*(neuralNetParameters.learningInputs.size()-1));

        for (int i = 0; i< neuralNetParameters.inputNodes; i++)
            neuralNetParameters.input.add(neuralNetParameters.learningInputs.get(neuralNetParameters.trainingLine).get(i));

        return neuralNetParameters.input.size();
    }

    public static int getTargetOutputs(NeuralNetParameters neuralNetParameters)
    {
        neuralNetParameters.target.clear();

        for (int i = 0; i< neuralNetParameters.outputNodes; i++)
            neuralNetParameters.target.add(neuralNetParameters.learningOutputs.get(neuralNetParameters.trainingLine).get(i));

        return neuralNetParameters.target.size();
    }
}
