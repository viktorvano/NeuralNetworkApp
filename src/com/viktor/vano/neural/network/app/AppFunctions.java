package com.viktor.vano.neural.network.app;

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
import java.util.Locale;

import static com.viktor.vano.neural.network.app.FFNN.GeneralFunctions.showVectorValues;
import static com.viktor.vano.neural.network.app.GUI.GUI.customPrompt;
import static com.viktor.vano.neural.network.app.Variables.*;

public class AppFunctions {
    public static void initializeLayout()
    {
        buttonFile = new Button("File");
        buttonFile.setLayoutX(stageWidth*0.05);
        buttonFile.setLayoutY(stageHeight*0.05);
        buttonFile.setOnAction(event -> {
            topologyFile = fileChooser.showOpenDialog(stageReference);
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
                    createNewNeuralNetwork();
                    createNewNeuralButtons();
                    createNewSlidersAndTextFields();
                }else if(neuralNetParameters != null && !filesOK)
                {
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
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Topology file", "topology*"));

        buttonTrain = new Button("Train");
        buttonTrain.setLayoutX(stageWidth*0.85);
        buttonTrain.setLayoutY(stageHeight*0.05);
        buttonTrain.setDisable(true);
        buttonTrain.setOnAction(event -> {
            if(neuralNetwork != null)
            {
                neuralNetwork.trainNeuralNetwork();
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
            if(neuralNetwork != null)
            {
                imaginationOfNN();
            }
        });
        pane.getChildren().add(buttonImagine);

        Timeline timelineRefresh = new Timeline(new KeyFrame(Duration.millis(250), event -> {
            buttonTrain.setDisable(neuralNetwork == null ||  neuralNetwork.isNetTraining() || update);
            buttonRandomRun.setDisable(neuralNetwork == null ||  neuralNetwork.isNetTraining() || update);
            buttonFile.setDisable(neuralNetwork != null && (neuralNetwork.isNetTraining() || update));
            buttonImagine.setDisable(neuralNetwork == null ||  neuralNetwork.isNetTraining() || update);

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

                for(int l = 0; l < neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1); l++)
                {
                    sliderOutputs.get(l).setValue(neuralNetwork.neuralNetParameters.result.get(l));
                }
            }

            if(updateInputSliders)
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
        sliderInputs = new ArrayList<>();
        sliderOutputs = new ArrayList<>();
        textFieldInputs = new ArrayList<>();
        textFieldOutputs = new ArrayList<>();
        for(int l = 0; l < neuralNetParameters.topology.get(0); l++)
        {
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
                sliderInputs.get(l).setLayoutX(0.05*stageWidth);
                sliderInputs.get(l).setLayoutY(0.24*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);

                textFieldInputs.get(l).setLayoutX(0.05*stageWidth);
                textFieldInputs.get(l).setLayoutY(0.28*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
            }

            for(int l = 0; l < neuralNetParameters.topology.get(neuralNetParameters.topology.size()-1); l++)
            {
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
        update = false;

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
    }

    public static void runCycleOfNN()
    {
        update = false;

        showVectorValues("Inputs:", neuralNetwork.neuralNetParameters.input);
        neuralNetwork.feedForward(neuralNetwork.neuralNetParameters.input);

        assert(neuralNetwork.neuralNetParameters.input.size() ==
                neuralNetwork.neuralNetParameters.topology.get(0));

        // Collect the net's actual results:
        neuralNetwork.getResults(neuralNetwork.neuralNetParameters.result);
        showVectorValues("Outputs: ", neuralNetwork.neuralNetParameters.result);

        update = true;
    }

    public static void imaginationOfNN()
    {
        update = false;
        updateInputSliders = false;

        try{
            neuralNetwork.neuralNetParameters.target.clear();
            for (Slider output : sliderOutputs)
            {
                neuralNetwork.neuralNetParameters.target.add((float)output.getValue());
            }

            showVectorValues("Feed outputs:", neuralNetwork.neuralNetParameters.target);
            neuralNetwork.neuralNetParameters.input.clear();
            neuralNetwork.neuralNetParameters.input = neuralNetwork.imagine(neuralNetwork.neuralNetParameters.target);
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
    }
}
