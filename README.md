# DiffusionModels

Dies ist die praktische Umsetzung meiner Masterarbeit "Erzeugung von Bildern und Videos aus Textbeschreibungen mittels Diffusionsmodellen". Sie beinhaltet Klassen sowie Skripte zum Training und zum Testen verschiedener Typen von Diffusionsmodellen. Alle Inhalte wurden in meiner [Masterarbeit](https://fh-swf.sciebo.de/s/8RXh0Mf5N5FKTs6) hinreichend beschrieben. Bei weiteren Fragen wenden Sie sich bitte an meine [Mail](mailto:schmidt.sebastian2@fh-swf.de) oder öffnen ein Issue im im [Git-Repository](https://github.com/sesch023/Diffusion).

## Installation:

Das Projekt wurde getestet auf dem [KI-Cluster](https://www.ki.fh-swf.de) der Fachhochschule Südwestfalen mit dem vorliegenden "Deep Learning and Datascience"-Image, welches Zugriff auf eine/die GPU(s) des Clusters bietet. Von anderen Umgebungen wird außer für außenstehende Nutzer abgeraten.

Das Projekt nutzt die folgenden Versionen und Kernpakete:

- Python (3.10.11) 
- PyTorch (2.0.0) 
- PyTorch Lightning (2.0.9) 
- torchmetrics (1.1.2) 
- torchinfo (1.8.0) 
- NumPy (1.23.5) 
- CLIP (1.0) 
- img2dataset (1.42.0)
- webdataset (0.2.48)

Diese und weitere nötige Pakete können mittels der "requirements.txt" installiert werden. Für das Training und den Test einiger Modelle müssen zusätzliche Basismodelle gedownloadet werden. Auch müssen ggf. Datensätze heruntergeladen und Datensatzpfade angepasst werden, falls nicht mit dem Cluster der Fachhochschule Südwestfalen gearbeitet wird. Links hierzu finden sich im Abschnitt Downloads.

## Aufbau des Projekts:

Alle Trainings- und andere relevante Dateien sind im Hauptverzeichnis "DiffusionModels" des Projekt untergebracht. Diese ausführbaren Python-Dateien Enden mit dem Suffix "Train". Zusätzlich findet sich in diesem Verzeichnis eine Configs.py, die verschiedene Konfigurationen beinhaltet in denen bei einigen Umgebungen Pfade angepasst werden müssen, falls Modelle oder Datensätze separat heruntergeladen wurden. Da Modelle und Datensätze auf dem Cluster im geteilten Ordner "archive" hinterlegt wurden, sollte in diesem Fall ein Download nicht nötig sein. Trotzdem werden alle nötigen Daten als separate Downloads hinterlegt.

Im Verzeichnis "test_runs" finden sich sechs Testskripte für die in der Masterarbeit beschriebenen Tests. Diese generieren eine definierbare Anzahl von Samples, geben die zugehörigen realen Beispiele aus und berechnen Metriken. Für die Berechnung der Precision und des Recalls wurde das Skript "improved_precision_recall.py" im Verzeichnis "test_runs/precision-and-recall-metric" verwendet. Zur Berechnung der finalen FID wurde das Programm "FullFidCalc.py" im Verzeichnis "test_runs/scripts" genutzt. Eine Berechnung der finalen FVD erfolgte über das zugehörige Skript im [StyleGAN-V-Repositiory](https://github.com/universome/stylegan-v) und die dort beschriebenen Programmbefehle.

Der Kern des Programms und alle relevante Klassen finden sich im Verzeichnis "DiffusionModules" im Hauptordner des Projekts. Dort ist in einem Verzeichnis "Legacy" ebenfalls alter Code hinterlegt, dieser ist jedoch nicht dokumentiert. 

## Modelle der Train-Dateien:

In diesem Abschnitt werden kurz die Modelle der Train-Dateien beschrieben. Dabei werden die relevantesten Modelle zunächst genannt. Einige Trainingsdateien wurden länger nicht erprobt und sind lediglich als Ergänzung hinterlegt. Diese werden hier als Legacy markiert. Eine Ausführung dieser sollte möglich sein, jedoch sind hier keine sinnvollen Ergebnisse versichert.

- CF10DiffusionTrain: Trainiert ein konditionales Diffusionsmodell für den CIFAR-10 Datensatz und einer linearen Schedule. Dieses Modell wurde in der Masterarbeit beschrieben.
    - Nötige separate Modelle: Upscaler-Modell
    - Nötige Datensätze: CIFAR-10-64
- CosDiffusionTrain: Trainiert ein konditionales Diffusionsmodell mit dem CC3M, CC12M und einer Cosine-Schedule. Es validiert mit dem MS COCO. Dieses Modell wurde in der Masterarbeit beschrieben.
    - Nötige separate Modelle: Upscaler-Modell
    - Nötige Datensätze: CC3M (Webdataset), CC12M (Webdataset), MS COCO (Webdataset)
- UpscalerTrain: Trainiert ein Embedding-konditionales Upscaler-Diffusionsmodell mit dem CC3M, CC12M und einer linearen Schedule. Es validiert mit dem MS COCO. Dieses Modell wurde in der Masterarbeit beschrieben.
    - Nötige separate Modelle: -
    - Nötige Datensätze: CC3M (Webdataset), CC12M (Webdataset), MS COCO (Webdataset)
- EmbVQGANTrain: Trainiert ein VQGAN mit unterstützenden Embeddings. Es trainiert mit dem CC3M sowie CC12M und validiert mit dem MS COCO. Dieses Modell wurde in der Masterarbeit beschrieben.
    - Nötige separate Modelle: -
    - Nötige Datensätze: CC3M (Webdataset), CC12M (Webdataset), MS COCO (Webdataset)
- LatentDiffusionTrain: Trainiert ein latentes Diffusionsmodell mit dem vorher beschriebenen VQGAN. Es trainiert mit dem CC3M sowie CC12M und validiert mit dem MS COCO. Dieses Modell wurde in der Masterarbeit beschrieben.
    - Nötige separate Modelle: VQGAN-Modell
    - Nötige Datensätze: CC3M (Webdataset), CC12M (Webdataset), MS COCO (Webdataset)
- SpatioTemporalDiffusionTrain: Trainiert den Spatiotemporal Decoder des Make-A-Video Systems. Abhängig der Konstante "skip_spatio" trainiert es erst mit dem CC3M und CC12M sowie validiert mit dem MS COCO oder es trainiert direkt mit dem WebVid10M. Ist "skip_spatio" false, so trainiert es zunächst bildlich für die im Trainer angegebene Zahl an Epochen. Dieses Modell wurde in der Masterarbeit beschrieben.
    - Nötige separate Modelle: -
    - Nötige Datensätze: CC3M (Webdataset), CC12M (Webdataset), MS COCO (Webdataset), WebVid10M

## Downloads-Modelle:

Hinweis: Vom Download des Upscaler-Modells kann theoretisch abgesehen werden, wenn im DiffusionTrainer ein anderer UpscalerMode als "UDM" übergeben wird. Dem USB-Stick der Abgabe liegen alle drei Modelle bei.

- Upscaler-Modell
- VQGAN-Modell
- CLIP-Translator-Modell

## Downloads-Datensätze:

Anmerkungen für den Download mit img2dataset: Der Standardbefehl für den Download fügt eine weiße Border zu Bildern einer nicht quadratischen Form hinzu. Dies ist für die Diffusion ungünstig. Die implementierten DataModules können mit nicht quadratischen Daten umgehen. Ein Download sollte daher entweder in originaler Auflösung oder mit der Option "keep_ratio" des Parameters "resize_mode" erfolgen. Der Download aller img2dataset Datensätze sollte mindestens mit der Größe 256 erfolgen!

- [CIFAR-10-64](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution)
- [MS COCO](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/mscoco.md)
- [CC3M (Webdataset)](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md)
- [CC12M (Webdataset)](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md)
- [WebVid-10M](https://github.com/m-bain/webvid)