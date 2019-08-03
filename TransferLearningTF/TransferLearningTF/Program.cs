using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace TransferLearningTF
{
    internal class Program
    {
        private static readonly string AssetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        private static readonly string TrainTagsTsv = Path.Combine(AssetsPath, "inputs-train", "data", "tags.tsv");
        private static readonly string PredictImageListTsv = Path.Combine(AssetsPath, "inputs-predict", "data", "image_list.tsv");
        private static readonly string TrainImagesFolder = Path.Combine(AssetsPath, "inputs-train", "data");
        private static readonly string PredictImagesFolder = Path.Combine(AssetsPath, "inputs-predict", "data");
        private static readonly string PredictSingleImage = Path.Combine(AssetsPath, "inputs-predict-single", "data", "toaster3.jpg");
        private static readonly string InceptionPb = Path.Combine(AssetsPath, "inputs-train", "inception", "tensorflow_inception_graph.pb");
        private static readonly string OutputImageClassifierZip = Path.Combine(AssetsPath, "outputs", "imageClassifier.zip");
        private const string LabelToKey = nameof(LabelToKey);
        private const string PredictedLabelValue = nameof(PredictedLabelValue);

        private static void Main()
        {
            var mlContext = new MLContext(1);
            var model = ReuseAndTuneInceptionModel(mlContext, TrainTagsTsv, InceptionPb);
            ClassifyImages(mlContext, PredictImageListTsv, PredictImagesFolder, model);
            ClassifySingleImage(mlContext, PredictSingleImage, model);
        }

        public static ITransformer ReuseAndTuneInceptionModel(MLContext mlContext, 
            string dataLocation,string inputModelLocation)
        {
            var data = mlContext.Data.LoadFromTextFile<ImageData>(dataLocation);
            var estimator = mlContext.Transforms.Conversion.MapValueToKey(LabelToKey, "Label")
                .Append(mlContext.Transforms.LoadImages("input", 
                    TrainImagesFolder, nameof(ImageData.ImagePath)))

                .Append(mlContext.Transforms.ResizeImages("input", 
                    InceptionSettings.ImageWidth, InceptionSettings.ImageHeight,
                    "input"))

                .Append(mlContext.Transforms.ExtractPixels("input", 
                    interleavePixelColors: InceptionSettings.ChannelsLast, 
                    offsetImage: InceptionSettings.Mean))

                .Append(mlContext.Model.LoadTensorFlowModel(inputModelLocation).
                    ScoreTensorFlowModel(new[] { "softmax2_pre_activation" }, 
                        new[] { "input" }, true))

                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(LabelToKey, 
                    "softmax2_pre_activation"))

                .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, 
                    "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            ConsoleWriteHeader("=============== Training classification model ===============");
            ITransformer model = estimator.Fit(data);
            var predictions = model.Transform(data);
            var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, 
                false, true);

            DisplayResults(imagePredictionData);

            var multiClassContext = mlContext.MulticlassClassification;
            var metrics = multiClassContext.Evaluate(predictions, LabelToKey);

            ConsoleWriteHeader("=============== Classification metrics ===============");
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: " +
                              $"{string.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString(CultureInfo.InvariantCulture)))}");
            return model;
        }

        public static void ClassifyImages(MLContext mlContext, string dataLocation, string imagesFolder, ITransformer model)
        {
            var imageData = ReadFromTsv(dataLocation, imagesFolder);
            var imageDataView = mlContext.Data.LoadFromEnumerable(imageData);
            var predictions = model.Transform(imageDataView);
            var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);
            
            ConsoleWriteHeader("=============== Making classifications ===============");
            DisplayResults(imagePredictionData);
        }

        public static void ClassifySingleImage(MLContext mlContext, string imagePath, ITransformer model)
        {
            var imageData = new ImageData
            {
                ImagePath = imagePath
            };

            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            ConsoleWriteHeader("=============== Making single image classification ===============");
            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} " +
                              $"predicted as: {prediction.PredictedLabelValue} " +
                              $"with score: {prediction.Score.Max()} ");
        }

        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
                .Select(line => line.Split('\t'))
                .Select(line => new ImageData
                {
                    ImagePath = Path.Combine(folder, line[0])
                });
        }

        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (var prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} " +
                                  $"predicted as: {prediction.PredictedLabelValue} " +
                                  $"with score: {prediction.Score.Max()} ");
            }
        }

        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new string('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const bool ChannelsLast = true;
        }
    }
}
