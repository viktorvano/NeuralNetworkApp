package com.viktor.vano.neural.network.app;

import com.sun.istack.internal.NotNull;
import com.viktor.vano.neural.network.app.FFNN.NeuralNetParameters;
import com.viktor.vano.neural.network.app.FFNN.NeuralNetwork;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.control.*;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.util.Duration;

import java.io.File;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;

import static com.viktor.vano.neural.network.app.FFNN.FileManagement.readOrCreateFile;
import static com.viktor.vano.neural.network.app.FFNN.GeneralFunctions.showVectorValues;
import static com.viktor.vano.neural.network.app.GUI.GUI.confirmationDialog;
import static com.viktor.vano.neural.network.app.GUI.GUI.customPrompt;
import static com.viktor.vano.neural.network.app.Variables.*;

public class AppFunctions {
    public static void initializeLayout()
    {
        createDirectoryIfNotExist("res");
        buttonFile = new Button("File");
        buttonFile.setLayoutX(stageWidth*0.05);
        buttonFile.setLayoutY(stageHeight*0.05);
        buttonFile.setOnAction(event -> {
            topologyFile = fileChooser.showOpenDialog(stageReference);
            fileChooser.setInitialDirectory(new File("res/"));
            if (topologyFile != null) {
                filesOK = true;
                System.out.println("File: " + topologyFile.getPath());
                String basePath = "";
                String fileNamingConvention = "";
                try{
                    String[] strings = topologyFile.getPath().split("topology_");

                    if(strings.length == 2 && strings[0].length() > 0)
                        basePath = strings[0];
                    else
                    {
                        filesOK = false;
                        customPrompt("File Chooser", "You have chosen an incorrect topology file: "
                                + topologyFile.getPath(), Alert.AlertType.WARNING);
                    }

                    strings = strings[1].split(".txt");
                    if(strings.length == 1 && strings[0].length() > 0)
                        fileNamingConvention = strings[0];
                    else
                    {
                        filesOK = false;
                        customPrompt("File Chooser", "Cannot determine the file naming convention from the topology file: "
                                + topologyFile.getPath(), Alert.AlertType.WARNING);
                    }
                }catch (Exception e)
                {
                    e.printStackTrace();
                    filesOK = false;
                    customPrompt("File Chooser", "Something went wrong picking a topology file: "
                            + topologyFile.getPath(), Alert.AlertType.WARNING);
                }

                if(topologyFile.getPath().length() > 50)
                    labelTopologyFile.setText("..." + topologyFile.getPath().substring(topologyFile.getPath().length()-50));
                else
                    labelTopologyFile.setText(topologyFile.getPath());

                trainingFile = new File(basePath + "training_" + fileNamingConvention + ".csv");
                if(!trainingFile.canRead())
                {
                    filesOK = false;
                    customPrompt("File Chooser", "Training file can not be read: "
                            + trainingFile.getPath(), Alert.AlertType.WARNING);
                }else
                {
                    loadLabels(trainingFile.getPath());
                }

                if(trainingFile.getPath().length() > 50)
                    labelTrainingFile.setText("..." + trainingFile.getPath().substring(trainingFile.getPath().length()-50));
                else
                    labelTrainingFile.setText(trainingFile.getPath());


                trainingStatusFile = new File(basePath + "trainingStatus_" + fileNamingConvention + ".txt");
                if(!trainingStatusFile.canRead())
                {
                    customPrompt("File Chooser", "Training status file does not exist: "
                            + trainingStatusFile.getPath(), Alert.AlertType.INFORMATION);
                }else
                {
                    if(trainingStatusFile.getPath().length() > 50)
                        labelTrainingStatusFile.setText("..." + trainingStatusFile.getPath().substring(trainingStatusFile.getPath().length()-50));
                    else
                        labelTrainingStatusFile.setText(trainingStatusFile.getPath());
                }

                weightsFile = new File(basePath + "weights_" + fileNamingConvention + ".dat");
                if(!weightsFile.canRead())
                {
                    customPrompt("File Chooser", "Weights file can not be read: "
                            + weightsFile.getPath() + "\nNew file will be created.", Alert.AlertType.INFORMATION);
                }

                if(weightsFile.getPath().length() > 50)
                    labelWeightsFile.setText("..." + weightsFile.getPath().substring(weightsFile.getPath().length()-50));
                else
                    labelWeightsFile.setText(weightsFile.getPath());

                if(neuralNetParameters == null && filesOK)
                {
                    enableActionButtons();
                    createNewNeuralNetwork();
                    createNewNeuralButtons();
                    createNewSlidersAndTextFields();
                }else if(neuralNetParameters != null && !filesOK)
                {
                    disableActionButtons();
                    removeOldChildren();
                }else if(neuralNetParameters != null)
                {
                    removeOldChildren();
                    createNewNeuralNetwork();
                    createNewNeuralButtons();
                    createNewSlidersAndTextFields();
                }
            }
        });
        pane.getChildren().add(buttonFile);

        labelTopologyFile = new Label("Please select a topology file.");
        labelTopologyFile.setFont(Font.font("Arial", 16));
        labelTopologyFile.setLayoutX(stageWidth*0.12);
        labelTopologyFile.setLayoutY(stageHeight*0.05);
        pane.getChildren().add(labelTopologyFile);

        labelTrainingFile = new Label("Training file not selected.");
        labelTrainingFile.setFont(Font.font("Arial", 16));
        labelTrainingFile.setLayoutX(stageWidth*0.12);
        labelTrainingFile.setLayoutY(stageHeight*0.08);
        pane.getChildren().add(labelTrainingFile);

        labelTrainingStatusFile = new Label("Status file not selected.");
        labelTrainingStatusFile.setFont(Font.font("Arial", 16));
        labelTrainingStatusFile.setLayoutX(stageWidth*0.12);
        labelTrainingStatusFile.setLayoutY(stageHeight*0.11);
        pane.getChildren().add(labelTrainingStatusFile);

        labelWeightsFile = new Label("Weights file not selected.");
        labelWeightsFile.setFont(Font.font("Arial", 16));
        labelWeightsFile.setLayoutX(stageWidth*0.12);
        labelWeightsFile.setLayoutY(stageHeight*0.14);
        pane.getChildren().add(labelWeightsFile);

        fileChooser = new FileChooser();
        fileChooser.setTitle("Open Topology File");
        fileChooser.setInitialDirectory(new File("res"));
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Topology file", "topology*"));

        buttonTrain = new Button("Train");
        buttonTrain.setLayoutX(stageWidth*0.85);
        buttonTrain.setLayoutY(stageHeight*0.05);
        buttonTrain.setDisable(true);
        buttonTrain.setOnAction(event -> {
            if(neuralNetwork != null
            && confirmationDialog(
                    "Training",
                    "This may take a while",
                    "Are you sure to start training?"))
            {
                pane.getChildren().add(progressBarTraining);
                buttonTrain.setDisable(true);
                buttonRandomRun.setDisable(true);
                buttonImagine.setDisable(true);
                neuralNetwork.trainNeuralNetwork();
                progressBarTraining.setProgress(neuralNetwork.getTrainingProgress());
            }
        });
        pane.getChildren().add(buttonTrain);

        buttonRandomRun = new Button("Random Run");
        buttonRandomRun.setLayoutX(stageWidth*0.85);
        buttonRandomRun.setLayoutY(stageHeight*0.10);
        buttonRandomRun.setDisable(true);
        buttonRandomRun.setOnAction(event -> {
            if(neuralNetwork != null)
            {
                runRandomCycleOfNN();
            }
        });
        pane.getChildren().add(buttonRandomRun);

        buttonImagine = new Button("Imagine");
        buttonImagine.setLayoutX(stageWidth*0.85);
        buttonImagine.setLayoutY(stageHeight*0.15);
        buttonImagine.setDisable(true);
        buttonImagine.setOnAction(event -> {
            if(neuralNetwork != null
            && confirmationDialog(
                    "Imagination",
                    "This may take a while",
                    "Are you sure to start imagination with current parameters?"))
            {
                pane.getChildren().add(progressBarTraining);
                buttonTrain.setDisable(true);
                buttonRandomRun.setDisable(true);
                buttonImagine.setDisable(true);
                imagination = new Imagination(0.0f, 1.0f);
                imagination.setName("Imagination Thread " + Math.round(Math.random()*1000.0));
                imagination.start();
                progressBarTraining.setProgress(neuralNetwork.getImaginationProgress());
            }
        });
        pane.getChildren().add(buttonImagine);

        progressBarTraining = new ProgressBar();
        progressBarTraining.setPrefWidth(stageWidth*0.8);
        progressBarTraining.setLayoutX(0.1*stageWidth);
        progressBarTraining.setLayoutY(0.01*stageHeight);

        Timeline timelineRefresh = new Timeline(new KeyFrame(Duration.millis(250), event -> {
            buttonFile.setDisable(neuralNetwork != null && (neuralNetwork.isNetTraining() || update));

            if(neuralNetwork != null)
            {
                if(neuralNetwork.isImaginationRunning())
                    progressBarTraining.setProgress(neuralNetwork.getImaginationProgress());

                if(neuralNetwork.isNetTraining())
                    progressBarTraining.setProgress(neuralNetwork.getTrainingProgress());
            }

            if(neuralNetwork != null
                    && !neuralNetwork.isImaginationRunning()
                    && neuralNetwork.getImaginationProgress() != 0.0f
                    && pane.getChildren().contains(progressBarTraining))
            {
                pane.getChildren().remove(progressBarTraining);
                buttonTrain.setDisable(false);
                buttonRandomRun.setDisable(false);
                buttonImagine.setDisable(false);
                customPrompt("Imagination",
                        "Imagination finished with " + neuralNetwork.getImaginationProgress()*100.0f + " % matching criteria.",
                        Alert.AlertType.INFORMATION);
            }

            if(neuralNetwork != null
                    && !neuralNetwork.isNetTraining()
                    && neuralNetwork.getTrainingProgress() != 0.0f
                    && pane.getChildren().contains(progressBarTraining))
            {
                pane.getChildren().remove(progressBarTraining);
                buttonTrain.setDisable(false);
                buttonRandomRun.setDisable(false);
                buttonImagine.setDisable(false);
                customPrompt("Training",
                        "Training finished with " + neuralNetwork.getTrainingProgress()*100.0f + " % matching criteria.",
                        Alert.AlertType.INFORMATION);
            }

            if(stageReference.getWidth() != stageWidth || stageReference.getHeight() != stageHeight)
            {
                stageWidth = (int)stageReference.getWidth();
                stageHeight = (int)stageReference.getHeight();
                updateLayoutPositions();
                System.out.println("Updated layout from timeline.");
            }

            if(update && buttonNeurons != null)
            {
                update = false;
                for (int i = 0; i < buttonNeurons.size(); i++)
                {
                    for(int l = 0; l < buttonNeurons.get(i).size(); l++)
                    {
                        buttonNeurons.get(i).get(l).setText(formatFloatToString4(neuralNetwork.getNeuronOutput(i,l)));
                        buttonNeurons.get(i).get(l).setStyle(colorStyle(neuralNetwork.getNeuronOutput(i,l)));
                    }
                }
            }

            if(updateInputSliders &&
               neuralNetwork.neuralNetParameters.input.size() == neuralNetParameters.topology.get(0))
            {
                updateInputSliders = false;
                for(int l = 0; l < neuralNetParameters.topology.get(0); l++)
                {
                    sliderInputs.get(l).setValue(neuralNetwork.neuralNetParameters.input.get(l));
                }
            }
        }));
        timelineRefresh.setCycleCount(Timeline.INDEFINITE);
        timelineRefresh.play();
    }

    private static void loadLabels(@NotNull String trainingFilePath)
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

            if(inputLabels == null)
                inputLabels = new ArrayList<>();
            else
                inputLabels.clear();

            inputLabels.addAll(Arrays.asList(inStrings));

            if(outputLabels == null)
                outputLabels = new ArrayList<>();
            else
                outputLabels.clear();

            outputLabels.addAll(Arrays.asList(outStrings));
        }catch (Exception e)
        {
            e.printStackTrace();
            customPrompt("File Chooser", "Training file can not read the first line properly: "
                    + trainingFile.getPath(), Alert.AlertType.WARNING);
        }
    }

    private static void disableActionButtons()
    {
        buttonTrain.setDisable(true);
        buttonRandomRun.setDisable(true);
        buttonImagine.setDisable(true);
    }

    private static void enableActionButtons()
    {
        buttonTrain.setDisable(false);
        buttonRandomRun.setDisable(false);
        buttonImagine.setDisable(false);
    }

    private static void removeOldChildren()
    {
        if(neuralNetwork != null && neuralNetwork.isNetTraining())
            neuralNetwork.stopTraining();

        for (ArrayList<Button> setOfNeuronButtons : buttonNeurons)
        {
            for (Button buttonNeuron : setOfNeuronButtons)
            {
                pane.getChildren().remove(buttonNeuron);
            }
            setOfNeuronButtons.clear();
        }
        buttonNeurons.clear();

        buttonNeurons = null;
        neuralNetwork = null;
        neuralNetParameters = null;

        for(Label in : labelInputs)
        {
            pane.getChildren().remove(in);
        }
        labelInputs.clear();

        for (Slider slider : sliderInputs)
        {
            pane.getChildren().remove(slider);
        }
        sliderInputs.clear();

        for (TextField textField : textFieldInputs)
        {
            pane.getChildren().remove(textField);
        }
        textFieldInputs.clear();

        for(Label out : labelOutputs)
        {
            pane.getChildren().remove(out);
        }
        labelOutputs.clear();

        for (Slider slider : sliderOutputs)
        {
            pane.getChildren().remove(slider);
        }
        sliderOutputs.clear();

        for (TextField textField : textFieldOutputs)
        {
            pane.getChildren().remove(textField);
        }
        textFieldOutputs.clear();
    }

    private static void createNewNeuralNetwork()
    {
        neuralNetParameters = new NeuralNetParameters(topologyFile.getPath(), trainingFile.getPath(),
                weightsFile.getPath(), trainingStatusFile.getPath(),
                0.1f,0.5f, 0.001f, 5000, 1000000);
        neuralNetwork = new NeuralNetwork(neuralNetParameters);
    }

    private static void createNewNeuralButtons()
    {
        buttonNeurons = new ArrayList<>();
        for (int i = 0; i < neuralNetParameters.topology.size(); i++)
        {
            buttonNeurons.add(new ArrayList<>());
            for(int l = 0; l < neuralNetParameters.topology.get(i); l++)
            {
                buttonNeurons.get(i).add(new Button(formatFloatToString4(0f)));
                buttonNeurons.get(i).get(l).setStyle(colorStyle(0f));
                buttonNeurons.get(i).get(l).setPrefSize(70, 40);
                buttonNeurons.get(i).get(l).setLayoutX(0.2*stageWidth +
                        (0.6*stageWidth / ((float)neuralNetParameters.topology.size())) * i);
                buttonNeurons.get(i).get(l).setLayoutY(0.26*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(i))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(i))) / 2) - bottomOffset);
                pane.getChildren().add(buttonNeurons.get(i).get(l));
            }
        }
    }

    private static void createNewSlidersAndTextFields()
    {
        neuralNetwork.neuralNetParameters.input.clear();
        neuralNetwork.neuralNetParameters.target.clear();
        neuralNetwork.neuralNetParameters.result.clear();
        labelInputs = new ArrayList<>();
        labelOutputs = new ArrayList<>();
        sliderInputs = new ArrayList<>();
        sliderOutputs = new ArrayList<>();
        textFieldInputs = new ArrayList<>();
        textFieldOutputs = new ArrayList<>();
        for(int l = 0; l < neuralNetParameters.topology.get(0); l++)
        {
            try
            {
                labelInputs.add(new Label(inputLabels.get(l)));
                labelInputs.get(l).setLayoutX(0.06*stageWidth);
                labelInputs.get(l).setLayoutY(0.2325*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
                pane.getChildren().add(labelInputs.get(l));
            }catch (Exception e)
            {
                e.printStackTrace();
            }

            sliderInputs.add(new Slider(-1.0, 1.0, 0));
            sliderInputs.get(l).setPrefWidth(110);
            sliderInputs.get(l).setLayoutX(0.05*stageWidth);
            sliderInputs.get(l).setLayoutY(0.25*stageHeight +
                    (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                    ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
            pane.getChildren().add(sliderInputs.get(l));

            textFieldInputs.add(new TextField("0.0"));
            textFieldInputs.get(l).setPrefSize(110, 40);
            textFieldInputs.get(l).setLayoutX(0.05*stageWidth);
            textFieldInputs.get(l).setLayoutY(0.27*stageHeight +
                    (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                    ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
            pane.getChildren().add(textFieldInputs.get(l));

            neuralNetwork.neuralNetParameters.input.add(0.0f);
            int finalL = l;
            sliderInputs.get(l).valueProperty().addListener(observable -> {
                neuralNetwork.neuralNetParameters.input.set(finalL, (float)sliderInputs.get(finalL).getValue());
                textFieldInputs.get(finalL).setText(String.format("%.3f", sliderInputs.get(finalL).getValue()).replace(",", "."));
                runCycleOfNN();
            });

            textFieldInputs.get(l).textProperty().addListener(observable -> {
                String text = textFieldInputs.get(finalL).getText();
                if(((text.contains("-") || text.contains("+")) && text.length() > 3)
                        || text.length() > 2)
                {
                    try{
                        float value = Float.parseFloat(text);
                        if(value > 1.0f)
                        {
                            value = 1.0f;
                        }else if(value < -1.0f)
                        {
                            value = -1.0f;
                        }

                        sliderInputs.get(finalL).setValue(value);
                        neuralNetwork.neuralNetParameters.input.set(finalL, value);
                        runCycleOfNN();
                    }catch (Exception e)
                    {
                        textFieldInputs.get(finalL).setText("");
                    }
                }
            });
        }

        for(int l = 0; l < neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1); l++)
        {
            try
            {
                labelOutputs.add(new Label(outputLabels.get(l)));
                labelOutputs.get(l).setLayoutX(0.81*stageWidth);
                labelOutputs.get(l).setLayoutY(0.2325*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) / 2) - bottomOffset);
                pane.getChildren().add(labelOutputs.get(l));
            }catch (Exception e)
            {
                e.printStackTrace();
            }

            sliderOutputs.add(new Slider(-1.0, 1.0, 0));
            sliderOutputs.get(l).setPrefWidth(110);
            sliderOutputs.get(l).setLayoutX(0.8*stageWidth);
            sliderOutputs.get(l).setLayoutY(0.25*stageHeight +
                    (0.75*stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) * l +
                    ((stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) / 2) - bottomOffset);
            pane.getChildren().add(sliderOutputs.get(l));

            textFieldOutputs.add(new TextField("0.0"));
            textFieldOutputs.get(l).setPrefSize(110, 40);
            textFieldOutputs.get(l).setLayoutX(0.8*stageWidth);
            textFieldOutputs.get(l).setLayoutY(0.27*stageHeight +
                    (0.75*stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) * l +
                    ((stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) / 2) - bottomOffset);
            pane.getChildren().add(textFieldOutputs.get(l));

            neuralNetwork.neuralNetParameters.result.add(0.0f);
            int finalL = l;
            sliderOutputs.get(l).valueProperty().addListener(observable -> {
                //neuralNetwork.neuralNetParameters.result.set(finalL, (float)sliderOutputs.get(finalL).getValue());
                textFieldOutputs.get(finalL).setText(String.format("%.3f", sliderOutputs.get(finalL).getValue()).replace(",", "."));
            });

            textFieldOutputs.get(l).textProperty().addListener(observable -> {
                String text = textFieldOutputs.get(finalL).getText();
                if(((text.contains("-") || text.contains("+")) && text.length() > 3)
                        || text.length() > 2)
                {
                    try{
                        float value = Float.parseFloat(text);
                        if(value > 1.0f)
                        {
                            value = 1.0f;
                        }else if(value < -1.0f)
                        {
                            value = -1.0f;
                        }

                        sliderOutputs.get(finalL).setValue(value);
                        //neuralNetwork.neuralNetParameters.result.set(finalL, value);
                    }catch (Exception e)
                    {
                        textFieldOutputs.get(finalL).setText("");
                    }
                }
            });
        }
    }

    public static String colorStyle(float value)
    {
        float color = value;
        if(value > 1.0f)
            value = 1.0f;
        else if(value < -1.0f)
            value = -1.0f;

        value *= 255.0f;
        boolean positive = true;
        if(value >= 0.0f)
        {
            positive=true;
        }else
        {
            positive = false;
            value = -value;
        }
        String hexString = Integer.toHexString((int)value);

        StringBuilder stringBuilder = new StringBuilder("-fx-background-color: #");
        String colorString = null;
        if (positive)
        {
            if (hexString.length()>1)
                stringBuilder.append("00" + hexString + "00;");
            else
                stringBuilder.append("000" + hexString + "00;");
        }else
        {
            if (hexString.length()>1)
                stringBuilder.append("0000" + hexString + ";");
            else
                stringBuilder.append("00000" + hexString + ";");
        }
        if(color > 0.6)
            stringBuilder.append(" -fx-text-fill: black;");
        else
            stringBuilder.append(" -fx-text-fill: white;");
        colorString = stringBuilder.toString();

        return colorString;
    }

    public static String formatFloatToString4(float number)
    {
        DecimalFormatSymbols formatSymbols = new DecimalFormatSymbols(Locale.getDefault());
        formatSymbols.setDecimalSeparator('.');
        return new DecimalFormat("##########.####", formatSymbols).format(number);
    }

    public static void updateLayoutPositions()
    {
        buttonFile.setLayoutX(stageWidth*0.05);
        buttonFile.setLayoutY(stageHeight*0.05);

        labelTopologyFile.setLayoutX(stageWidth*0.12);
        labelTopologyFile.setLayoutY(stageHeight*0.05);

        labelTrainingFile.setLayoutX(stageWidth*0.12);
        labelTrainingFile.setLayoutY(stageHeight*0.08);

        labelTrainingStatusFile.setLayoutX(stageWidth*0.12);
        labelTrainingStatusFile.setLayoutY(stageHeight*0.11);

        labelWeightsFile.setLayoutX(stageWidth*0.12);
        labelWeightsFile.setLayoutY(stageHeight*0.14);

        buttonTrain.setLayoutX(stageWidth*0.85);
        buttonTrain.setLayoutY(stageHeight*0.05);

        buttonRandomRun.setLayoutX(stageWidth*0.85);
        buttonRandomRun.setLayoutY(stageHeight*0.10);

        buttonImagine.setLayoutX(stageWidth*0.85);
        buttonImagine.setLayoutY(stageHeight*0.15);

        progressBarTraining.setPrefWidth(0.8*stageWidth);
        progressBarTraining.setLayoutX(0.1*stageWidth);
        progressBarTraining.setLayoutY(0.01*stageHeight);

        if(neuralNetParameters != null && filesOK)
        {
            for (int i = 0; i < neuralNetParameters.topology.size(); i++)
            {
                for(int l = 0; l < neuralNetParameters.topology.get(i); l++)
                {
                    buttonNeurons.get(i).get(l).setLayoutX(0.2*stageWidth +
                            (0.6*stageWidth / ((float)neuralNetParameters.topology.size())) * i);
                    buttonNeurons.get(i).get(l).setLayoutY(0.26*stageHeight +
                            (0.75*stageHeight / ((float)neuralNetParameters.topology.get(i))) * l +
                            ((stageHeight / ((float)neuralNetParameters.topology.get(i))) / 2) - bottomOffset);
                }
            }

            for(int l = 0; l < neuralNetParameters.topology.get(0); l++)
            {
                labelInputs.get(l).setLayoutX(0.06*stageWidth);
                labelInputs.get(l).setLayoutY(0.2325*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);

                sliderInputs.get(l).setLayoutX(0.05*stageWidth);
                sliderInputs.get(l).setLayoutY(0.25*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);

                textFieldInputs.get(l).setLayoutX(0.05*stageWidth);
                textFieldInputs.get(l).setLayoutY(0.27*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
            }

            for(int l = 0; l < neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1); l++)
            {
                labelOutputs.get(l).setLayoutX(0.81*stageWidth);
                labelOutputs.get(l).setLayoutY(0.2325*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) / 2) - bottomOffset);

                sliderOutputs.get(l).setLayoutX(0.8*stageWidth);
                sliderOutputs.get(l).setLayoutY(0.25*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) / 2) - bottomOffset);

                textFieldOutputs.get(l).setLayoutX(0.8*stageWidth);
                textFieldOutputs.get(l).setLayoutY(0.27*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1))) / 2) - bottomOffset);
            }
        }
    }

    public static void runRandomCycleOfNN()
    {
        disableActionButtons();
        update = true;
        updateInputSliders = true;

        neuralNetwork.neuralNetParameters.input.clear();
        for(int i = 0; i < neuralNetwork.neuralNetParameters.topology.get(0); i++)
        {
            float randomValue = (float)Math.random();
            neuralNetwork.neuralNetParameters.input.add(randomValue);
            sliderInputs.get(i).setValue(randomValue);
            textFieldInputs.get(i).setText(String.valueOf(randomValue));
        }
        showVectorValues("Inputs:", neuralNetwork.neuralNetParameters.input);
        neuralNetwork.feedForward(neuralNetwork.neuralNetParameters.input);

        assert(neuralNetwork.neuralNetParameters.input.size() ==
                neuralNetwork.neuralNetParameters.topology.get(0));

        // Collect the net's actual results:
        neuralNetwork.getResults(neuralNetwork.neuralNetParameters.result);
        showVectorValues("Outputs: ", neuralNetwork.neuralNetParameters.result);

        update = true;
        updateInputSliders = true;
        enableActionButtons();
    }

    public static void runCycleOfNN()
    {
        disableActionButtons();
        update = true;
        updateInputSliders = true;

        showVectorValues("Inputs:", neuralNetwork.neuralNetParameters.input);
        neuralNetwork.feedForward(neuralNetwork.neuralNetParameters.input);

        assert(neuralNetwork.neuralNetParameters.input.size() ==
                neuralNetwork.neuralNetParameters.topology.get(0));

        // Collect the net's actual results:
        neuralNetwork.getResults(neuralNetwork.neuralNetParameters.result);
        showVectorValues("Outputs: ", neuralNetwork.neuralNetParameters.result);

        update = true;
        updateInputSliders = true;
        enableActionButtons();
    }

    public static class Imagination extends Thread
    {
        private float minInValue, maxInValue;
        public Imagination(float minInValue, float maxInValue)
        {
            this.minInValue = minInValue;
            this.maxInValue = maxInValue;
        }

        @Override
        public void run()
        {
            disableActionButtons();
            update = true;
            updateInputSliders = true;

            try{
                final ArrayList<Float> dreamTarget = new ArrayList<>();
                dreamTarget.clear();
                for (Slider output : sliderOutputs)
                {
                    dreamTarget.add((float)output.getValue());
                }

                showVectorValues("Feed outputs:", dreamTarget);
                neuralNetwork.neuralNetParameters.input.clear();
                neuralNetwork.neuralNetParameters.input = neuralNetwork.
                        imagine(dreamTarget, this.minInValue, this.maxInValue);
                neuralNetwork.feedForward(neuralNetwork.neuralNetParameters.input);

                assert(neuralNetwork.neuralNetParameters.input.size() ==
                        neuralNetwork.neuralNetParameters.topology.get(0));

                // Collect the net's actual results:
                neuralNetwork.getResults(neuralNetwork.neuralNetParameters.result);
                showVectorValues("Outputs: ", neuralNetwork.neuralNetParameters.result);
            }catch (Exception e)
            {
                e.printStackTrace();
            }

            update = true;
            updateInputSliders = true;
            enableActionButtons();
        }
    }

    public static void createDirectoryIfNotExist(String directoryName)
    {
        File file = new File(directoryName);
        if(file.mkdir())
            System.out.println("New directory \"" + directoryName + "\" was created.");
        else
            System.out.println("Directory \"" + directoryName + "\" already exists.");
    }
}
