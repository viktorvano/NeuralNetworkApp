package com.viktor.vano.neural.network.app.FFNN;

import com.sun.istack.internal.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.IntStream;

import static FFNN.FileManagement.*;

public class Weights {
    public static void pushZerosToWeights(NeuralNetParameters neuralNetParameters)
    {
        int index, NumberOfWeights = 0;
        int topologySize = neuralNetParameters.topology.size();

        for (index = 0; index < topologySize - 1; index++)
        {
            NumberOfWeights += (neuralNetParameters.topology.get(index) + 1)* neuralNetParameters.topology.get(index + 1);
        }

        neuralNetParameters.weights = new ArrayList<>(Arrays.asList(new Float[NumberOfWeights]));
    }

    public static void pushZerosToLearningTable(NeuralNetParameters neuralNetParameters)
    {
        ArrayList<Float> InputRow = new ArrayList<>();
        ArrayList<Float> OutputRow = new ArrayList<>();
        int row, column;

        neuralNetParameters.learningInputs.clear();
        for (row = 0; row < neuralNetParameters.inputNodes; row++)
        {
            InputRow.add(0.0f);
        }
        for (column = 0; column < neuralNetParameters.trainingSamplesCount; column++)
        {
            neuralNetParameters.learningInputs.add(InputRow);
        }

        neuralNetParameters.learningOutputs.clear();
        for (row = 0; row < neuralNetParameters.outputNodes; row++)
        {
            OutputRow.add(0.0f);
        }
        for (column = 0; column < neuralNetParameters.trainingSamplesCount; column++)
        {
            neuralNetParameters.learningOutputs.add(OutputRow);
        }
    }

    public static void getTrainingDataCount(NeuralNetParameters neuralNetParameters)
    {
        ArrayList<String> fileContent = readOrCreateFile(neuralNetParameters.trainingFilePath);

        if(fileContent.size()==0 || fileContent==null)
        {
            System.out.println("Cannot open " + neuralNetParameters.trainingFilePath);
            System.exit(-5);
        }

        neuralNetParameters.trainingSamplesCount = fileContent.size() - 1;
    }

    public static void loadTopology(NeuralNetParameters neuralNetParameters)
    {
        ArrayList<String> fileContent = readOrCreateFile(neuralNetParameters.topologyFilePath);

        if(fileContent.size() == 0 || fileContent == null)
        {
            System.out.println("Cannot open " + neuralNetParameters.topologyFilePath);
            System.exit(-7);
        }

        for(int i = 0; i < fileContent.size(); i++)
        {
            String numberString = new String();
            for(int x = 0; x < fileContent.get(i).length(); x++)
            {
                char c = fileContent.get(i).charAt(x);
                if(c >= '0' && c <= '9')
                {
                    numberString += c;
                }
            }

            if(numberString != null && numberString.length() != 0)
            {
                neuralNetParameters.topology.add(Integer.parseInt(numberString));
                neuralNetParameters.inputNodes = neuralNetParameters.topology.get(0);
                neuralNetParameters.outputNodes = neuralNetParameters.topology.get(neuralNetParameters.topology.size() - 1);
                getTrainingDataCount(neuralNetParameters);
                pushZerosToLearningTable(neuralNetParameters);
                pushZerosToWeights(neuralNetParameters);
            }
        }
    }


    public static void loadTrainingDataFromFile(NeuralNetParameters neuralNetParameters)
    {
        ArrayList<String> fileContent = readOrCreateFile(neuralNetParameters.trainingFilePath);

        if(fileContent.size() == 0 || fileContent == null)
        {
            System.out.println("Cannot open " + neuralNetParameters.trainingFilePath);
            System.exit(-8);
        }

        for(int fileLine = 0; fileLine < fileContent.size(); fileLine++)
        {
            if(fileContent.get(fileLine).contains(";;"))
                fileContent.set(fileLine, fileContent.get(fileLine).replace(';', '\t'));
            else if(fileContent.get(fileLine).contains(",,"))
                fileContent.set(fileLine, fileContent.get(fileLine).replace(',', '\t'));

            if(fileContent.get(fileLine).contains(" "))
                fileContent.set(fileLine, fileContent.get(fileLine).replace(" ", ""));
        }

        try {
            String[] strings = fileContent.get(0).split("\t\t");
            String inputLabels = strings[0];
            String outputLabels = strings[1];

            String[] inputs = inputLabels.split("\t");
            String[] outputs = outputLabels.split("\t");

            neuralNetParameters.trainData.InputLabels.addAll(Arrays.asList(inputs));
            neuralNetParameters.trainData.OutputLabels.addAll(Arrays.asList(outputs));
        }catch (Exception e)
        {
            e.printStackTrace();
        }


        IntStream.range(1, fileContent.size()).forEach(
                i -> {
                    ArrayList<Float> inputLine = new ArrayList<>();
                    ArrayList<Float> outputLine = new ArrayList<>();

                    String[] strings = fileContent.get(i).split("\t\t");
                    String inputLabels = strings[0];
                    String outputLabels = strings[1];

                    String[] inputs = inputLabels.split("\t");
                    String[] outputs = outputLabels.split("\t");

                    for (String inValue : inputs)
                    {
                        inputLine.add(Float.parseFloat(inValue));
                    }

                    for (String outValue : outputs)
                    {
                        outputLine.add(Float.parseFloat(outValue));
                    }

                    neuralNetParameters.learningInputs.set(i-1, inputLine);
                    neuralNetParameters.learningOutputs.set(i-1, outputLine);
                });

        System.out.println("learningInputs: " + neuralNetParameters.learningInputs);
        System.out.println("learningOutputs: " + neuralNetParameters.learningOutputs);
    }

    public static int getNumberOfWeightsFromFile(NeuralNetParameters neuralNetParameters)
    {
        int number_of_weights = 0;

        ArrayList<String> fileContent = readOrCreateFile(neuralNetParameters.weightsFilePath);

        for (int i = 0; i < fileContent.size(); i++)
        {
            if(fileContent.get(i).length()!=0)
                number_of_weights++;
        }

        return number_of_weights;
    }

    public static void allocateNewWeights(@NotNull ArrayList<Integer> topology, @NotNull ArrayList<Float> weights)
    {
        int index, NumberOfWeights = 0;
        int topologySize = topology.size();

        for (index = 0; index < topologySize - 1; index++)
        {
            NumberOfWeights += (topology.get(index) + 1)*topology.get(index + 1);
        }

        weights = new ArrayList<>(Arrays.asList(new Float[NumberOfWeights]));
    }
}
