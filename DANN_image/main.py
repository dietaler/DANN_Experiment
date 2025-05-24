import torch
import train
# import mnist
# import mnistm
import model
import cicids2017
import cicids2018

def main():
    source_train_loader = cicids2017.train_loader_2017
    target_train_loader = cicids2018.train_loader_2018
    # source_train_loader = cicids2018.train_loader_2018
    # target_train_loader = cicids2017.train_loader_2017

    if torch.cuda.is_available():
        encoder_source_only = model.Extractor().cuda()
        classifier_source_only = model.Classifier().cuda()

        encoder_dann = model.Extractor().cuda()
        classifier_dann = model.Classifier().cuda()
        discriminator_dann = model.Discriminator().cuda()

        train.source_only(encoder_source_only, classifier_source_only, source_train_loader, target_train_loader)
        train.dann(encoder_dann, classifier_dann, discriminator_dann, source_train_loader, target_train_loader)
    else:
        print("No GPUs available.")


if __name__ == "__main__":
    main()
