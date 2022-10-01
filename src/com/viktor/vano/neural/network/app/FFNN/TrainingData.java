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

    public void loadLabels(@NotNull String trainingFilePath)
    {
        ArrayList<String> fileContent = readOrCreateFile(trainingFilePath);
        String firstLine;

        try
        {
            firstLine = fileContent.get(0);

            if(firstLine.contains(";;"))
                firstLine = firstLine.replace(';', '\t');
            else if(firstLine.contains(",,"))
                firstLine = firstLine.replace(',', '\t');

            String[] strings = firstLine.split("\t\t");
            String[] inStrings, outStrings;
            inStrings = strings[0].split("\t");
            outStrings = strings[1].split("\t");

            neuralNetParameters.trainData.inputLabels.clear();
            neuralNetParameters.trainData.inputLabels.addAll(Arrays.asList(inStrings));

            neuralNetParameters.trainData.outputLabels.clear();
            neuralNetParameters.trainData.outputLabels.addAll(Arrays.asList(outStrings));
        }catch (Exception e)
        {
            e.printStackTrace();
            customPrompt("File Chooser", "Training file can not read the first line properly: "
                    + trainingFile.getPath(), Alert.AlertType.WARNING);
        }
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
