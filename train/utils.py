import torch
import torch.nn as nn
import torch.nn.functional as F

# Binary Cross Entropy loss with logits
loss_object = nn.BCEWithLogitsLoss()

def generator_loss(disc_generated_output, gen_output, target):
    # GAN loss
    gan_loss = loss_object(disc_generated_output, torch.ones_like(disc_generated_output))

    # Mean absolute error (L1 loss)
    l1_loss = F.l1_loss(gen_output, target)

    # Euclidean (L2) loss
    r = target - gen_output
    l2_loss = torch.norm(r, p=2)

    # Total generator loss
    total_gen_loss = gan_loss + 100*l1_loss + l2_loss

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    # Real loss
    real_loss = loss_object(disc_real_output, torch.ones_like(disc_real_output))

    # Generated loss
    generated_loss = loss_object(disc_generated_output, torch.zeros_like(disc_generated_output))

    # Total discriminator loss
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

