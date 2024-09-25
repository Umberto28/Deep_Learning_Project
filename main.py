import argparse

from training import train, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--resume_iam', '-ri', action="store_true",
                        help="resume training")
    parser.add_argument('--resume', '-r', action="store_false",
                        help="resume training with no_augmented dataset")
    parser.add_argument('--model', '-m', choices=['resnet', 'vit'], default='resnet',
                        help="set which model to use / train, either 'resnet' or 'vit'")
    parser.add_argument('--aug', '-a', choices=['aug', 'no_aug'], default='no_aug',
                        help="set which dataset to use, either 'augmented' or 'no_augmented'")
    parser.add_argument('--split', '-s', default=0,
                        help="which train/val/test split to load")
    parser.add_argument('--weighted_loss', '-wl', action="store_true",
                        help="either using a weighted CrossEntropyLoss or not")
    parser.add_argument('--freeze', '-brr', action="store_true",
                        help="freeze all layers for training but the head")
    parser.add_argument('--explain', '-ex', action="store_true",
                        help="print explainer SHAP results")     
    parser.add_argument('--test', '-t', action="store_true",
                        help="only testing mode")                

    args = parser.parse_args()
    print(args)

    if not args.test: 
        train(args)
    test(args, explain=args.explain)

