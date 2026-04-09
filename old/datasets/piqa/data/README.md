## Physical Interaction: Question Answering

* [train.jsonl](train.jsonl)
* [train-labels.lst](train-labels.lst)
* [valid.jsonl](valid.jsonl)
* [valid-labels.lst](valid-labels.lst)
* [tests.jsonl](tests.jsonl)

There are two files for each split:  `X.jsonl` and `X-labels.lst`.  Lines are aligned.  Each item in the `jsonl` takes the form:
```
{
"goal": "How do I ready a guinea pig cage for it's new occupants?", 
"sol1": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.", 
"sol2": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."
}
```

and each line in the `.lst` is a `0` or `1`.  Pretty simple.

---

FAQ:  

1. **Some of these "goals" are one word/item?** Yeah, in those cases, the goal is often how to use that item or what you can use that item for.
2. **Some of these are definitions?** True
3. **Paper** [https://arxiv.org/abs/1911.11641](https://arxiv.org/abs/1911.11641)
4. **License** [Academic Free License ("AFL") v. 3.0](https://opensource.org/licenses/AFL-3.0)
