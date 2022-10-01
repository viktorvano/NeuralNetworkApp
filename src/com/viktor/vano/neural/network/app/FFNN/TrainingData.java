package com.viktor.vano.neural.network.app.FFNN;

import com.sun.istack.internal.NotNull;
import javafx.scene.control.Alert;

import java.util.ArrayList;
import java.util.Arrays;

import static com.viktor.vano.neural.network.app.FFNN.FileManagement.readOrCreateFile;
import static com.viktor.vano.neural.network.app.GUI.GUI.customPrompt;
import static com.viktor.vano.neural.network.app.Variables.neuralNetParameters;
import static com.viktor.vano.neural.network.app.Variables.trainingFile;

public class TrainingData {
    public ArrayList<String> inputLabels, outputLabels;

    public TrainingData()
    {
        inputLabels = new ArrayList<>();
        outputLabels = new ArrayList<>();
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
