from .mscmr import build, build_ACDC


def build_dataset(image_set, args):
    if args.dataset == 'MSCMR':
        return build(image_set, args)

    if args.dataset == 'ACDC':
        return build_ACDC(image_set, args)

    raise ValueError(f'dataset {args.dataset} not supported')
