from torch.utils.data import DataLoader
from utils.data.data_augment import data_aug


def get_dataloader(args, traindata,
        testdata,
        forgottenTrainata,
        resttraindata,
        thirdPartyData,):

    traindata = data_aug(traindata, args, mode="train")
    forgottenTrainata = data_aug(forgottenTrainata, args, mode="train")
    resttraindata = data_aug(resttraindata, args, mode="train")
    thirdPartyData = data_aug(thirdPartyData, args, mode="train")
    testdata = data_aug(testdata, args, mode="test")

    trainloader = DataLoader(
        traindata, batch_size=args.batchsize, shuffle=True, num_workers=8
    )

    forgottenTrainloader = DataLoader(
        forgottenTrainata,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    resttraindataloader = DataLoader(
        resttraindata,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    thirdPartyDataloader = DataLoader(
        thirdPartyData,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    testloader = DataLoader(
        testdata, batch_size=args.batchsize, shuffle=False, num_workers=8
    )

    return (
        trainloader,
        forgottenTrainloader,
        resttraindataloader,
        testloader,
        thirdPartyDataloader,
    )
