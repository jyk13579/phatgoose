#!/bin/bash

if [ "$1" = "train_lora" ]; then

    bash colm/experiments/bash_scripts/train_single_task_loralinear.sh -exp_name P3Socialiqa_t5xl_lora -dataset P3SOCIALIQA -extra_bindings 'P/TRAIN/Trainer.gradient_accumulation_factor=32';
    #avoid saving to GCP
    # bash colm/experiments/bash_scripts/train_single_task_loralinear.sh -exp_name P3Socialiqa_t5xl_lora -dataset P3SOCIALIQA -extra_bindings 'MOMA/save_weights.should_save_to_gcp=False P/TRAIN/Trainer.gradient_accumulation_factor=32';

elif [ "$1" = "convert_moe" ]; then
    python scripts/manipulations.py --gin_bindings 'put_index_to_lora.path="P3Socialiqa_t5xl_lora"' 'put_index_to_lora.out_path="datasets_concatenated/P3Socialiqa_t5xl_lora"' 'func_caller.func=@put_index_to_lora'

elif [ "$1" = "train_gate" ]; then
    bash colm/experiments/bash_scripts/train_gate.sh -exp_name datasets_concatenated/P3Socialiqa_t5xl_lora_inpgatetrainnogumbel -dataset P3SOCIALIQA -old_exp_name datasets_concatenated/P3Socialiqa_t5xl_lora -extra_bindings 'main.logging_backend=None P/TRAIN/Trainer.gradient_accumulation_factor=32';

elif [ "$1" = "convert_routing_vector" ]; then
    python scripts/manipulations.py --gin_bindings 'use_input_gate_as_router.path="datasets_concatenated/P3Socialiqa_t5xl_lora_inpgatetrainnogumbel"' 'func_caller.func=@use_input_gate_as_router';

elif [ "$1" = "concatenate_all_moe" ]; then
    python scripts/concatenate.py --gin_bindings 'run_concatenate.print_commands=False' 'run_concatenate.out_path="FullCompleteA2inpgatetrainnogumbel_t5xl_lora_concatenated"' 'func_caller.func=@run_concatenate' 'run_concatenate.suffix="t5xl_lora_inpgatetrainnogumbel"' 'run_concatenate.datasets="Full"'


elif [ "$1" = "eval_PHATGOOSE" ]; then
    bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name P3_Phatgoose -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/P3APPREVIEWS/EVAL", "D/P3TREC/EVAL", "D/P3MULTINEWS/EVAL", "D/P3IMDB/EVAL", "D/P3ADVERSARIALQA/EVAL", "D/P3CNNDAILYMAIL/EVAL", "D/P3DBPEDIA14/EVAL", "D/P3QUAIL/EVAL","D/P3DREAM/EVAL","D/P3YELP/EVAL"] M/MODEL/FFNExperts.topk_value=2 M/MODEL/FFNExperts.normalize_topk=True M/MODEL/ENCODER/ExposeHidden.reduction_method=None M/MODEL/DECODER/ExposeHidden.reduction_method=None P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText(), @RoutingDistribution()] WriteOutputText.save_dir="exp_out/FLAN_Phatgoose/output_text" RoutingDistribution.save_dir="exp_out/FLAN_Phatgoose/routing_distribution"'

    ## Big Bench hard
    # 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"]
    ## P3 held-out
    # "D/P3STORYCLOZE/EVAL" => local data path
    # "D/P3RTE/EVAL", "D/P3HSWAG/EVAL", "D/P3COPA/EVAL", "D/P3WIC/EVAL", "D/P3WINOGRANDE/EVAL", "D/P3CB/EVAL", "D/P3ANLI/R1/EVAL", "D/P3ANLI/R2/EVAL", "D/P3ANLI/R3/EVAL", "D/P3WSCFIXED/EVAL"
    ## P3 held-in
    #"D/P3AGNEWS/EVAL", "D/P3AMAZONPOLARITY/EVAL", "D/P3COSMOSQA/EVAL", "D/P3SAMSUM/EVAL", "D/P3QUARTZ/EVAL", "D/P3ROPES/EVAL", "D/P3WIKIBIO/EVAL", "D/P3PAWS/EVAL", "D/P3WIKIQA/EVAL", "D/P3SOCIALIQA/EVAL", "D/P3QASC/EVAL", "D/P3QUAIL/EVAL", "D/P3DREAM/EVAL", "D/P3WIQA/EVAL", "D/P3QUAREL/EVAL", "D/P3SCIQ/EVAL", "D/P3QUOREF/EVAL", "D/P3DUORC/EVAL", "D/P3ROTTENTOMATOES/EVAL", "D/P3YELP/EVAL", "D/P3COMMONGEN/EVAL", "D/P3GIGAWORD/EVAL", "D/P3XSUM/EVAL", "D/P3MRPC/EVAL", "D/P3QQP/EVAL", "D/P3COMMONSENSEQA/EVAL", "D/P3COSE/EVAL", "D/P3WIKIHOP/EVAL", "D/P3HOTPOTQA/EVAL", "D/P3APPREVIEWS/EVAL", "D/P3TREC/EVAL", "D/P3MULTINEWS/EVAL", "D/P3IMDB/EVAL", "D/P3ADVERSARIALQA/EVAL", "D/P3CNNDAILYMAIL/EVAL", "D/P3DBPEDIA14/EVAL"

elif [ "$1" = "eval_multitask" ]; then
    bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name flan_t5_xl -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"] P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/flan_t5_xl/output_text" M/MODEL/hf_torch_model.model_name_or_path="google/flan-t5-xl" M/MODEL/Model.init_moma_calls=[]'
    
elif [ "$1" = "eval_single_expert" ]; then
    bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name datasets_concatenated/P3Socialiqa_t5xl_lora -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"] M/MODEL/ENCODER/ExposeHidden.reduction_method="masked_mean" M/MODEL/DECODER/ExposeHidden.reduction_method="mean" P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/datasets_concatenated/P3Socialiqa_t5xl_lora/output_text"'

elif [ "$1" = "eval_retrieval" ]; then
    bash colm/experiments/bash_scripts/retriever.sh -dataset_setting Full -extra_bindings 'main.procedure_exec_order=["P/EVALUATE/BBH"] P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/FullCompleteansretrieval_t5xl_lora_concatenated/output_text"'

elif [ "$1" = "eval_merged_experts" ]; then
    bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name FLAN_MergedExperts -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"] M/MODEL/ENCODER/ExposeHidden.reduction_method="masked_mean" M/MODEL/DECODER/ExposeHidden.reduction_method="mean" P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/FLAN_MergedExperts/output_text"'

elif [ "$1" = "eval_average_activation" ]; then
    bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name FLAN_AverageActivation -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"] M/MODEL/FFNExperts.topk_value=2 M/MODEL/FFNExperts.normalize_topk=True M/MODEL/ENCODER/ExposeHidden.reduction_method=None M/MODEL/DECODER/ExposeHidden.reduction_method=None P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText(), @RoutingDistribution()] WriteOutputText.save_dir="exp_out/FLAN_AverageActivation/output_text" RoutingDistribution.save_dir="exp_out/FLAN_AverageActivation/routing_distribution"'

else
	echo "Invalid Option Selected"
fi
