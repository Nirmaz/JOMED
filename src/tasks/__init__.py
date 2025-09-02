from . import base, vqa_rad,  cr_qa_breast, cr_qa_retina, cr_qa_derma,  multi_label

AVAILABLE_TASKS = {m.__name__.split(".")[-1]: m for m in [base, vqa_rad,  cr_qa_breast, cr_qa_retina, cr_qa_derma, multi_label]}


def get_task(opt, tokenizer):
    if opt.task not in AVAILABLE_TASKS:
        raise ValueError(f"{opt.task} not recognised")
    task_module = AVAILABLE_TASKS[opt.task]
    return task_module.Task(opt, tokenizer)
