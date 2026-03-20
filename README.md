# Facial Recognition Fairness Audit

This project is about building a simple facial verification system and checking how it performs on different types of people.

## What I did

- Used UTKFace dataset
- Built a model using ResNet18
- Trained using cosine similarity loss
- Compared two face images to check if they are same
- Calculated FAR and FRR for evaluation

## Fairness Audit

I divided the data based on:
- Gender
- Age groups
- Skin tone (based on dataset labels)

Then I checked how the model performs for each group.

## Results

The model did not perform equally for all groups. Some groups had higher false reject rates.

## Mitigation

I changed the threshold value to reduce the difference between groups. It improved fairness slightly but reduced accuracy a bit.

## Conclusion

The model still has some bias and is not suitable for real-world high-risk use.

## Note

Dataset is not uploaded due to size limits.