
def create_model(opt):

    if opt.model == "unrealego2_pose_qa_df":
        from .unrealego2_pose_qa_depth_model import UnrealEgo2PoseQADepthModel
        model = UnrealEgo2PoseQADepthModel()
    else:
        raise ValueError('Model [%s] not recognized.' % opt.model)

    model.initialize(opt)

    return model