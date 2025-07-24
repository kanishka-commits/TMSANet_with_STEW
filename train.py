import copy
import os
import time
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from torch import nn

import config
from util import data_augmentation, GradCAM, plot_gradcam_on_eeg


def train_evaluation(
    model: nn.Module,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    save_path,
    epochs=config.epochs,
    patience=30
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    best_acc = 0
    avg_acc = 0
    best_model = None
    best_kappa = 0
    early_stop_counter = 0

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    os.makedirs(save_path, exist_ok=True)
    log_writer = open(os.path.join(save_path, 'log.txt'), 'w')

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_predicted, train_actual = [], []

        # TRAIN LOOP
        for train_data, train_labels in train_loader:
            train_data = train_data.squeeze(1) if train_data.dim() == 4 else train_data
            train_data, train_labels = train_data.to(device).float(), train_labels.to(device).long()

            # Augment data
            aug_data, aug_labels = data_augmentation(train_data, train_labels)
            aug_data = aug_data.squeeze(1) if aug_data.dim() == 4 else aug_data

            train_data = torch.cat((train_data, aug_data), dim=0).float()
            train_labels = torch.cat((train_labels, aug_labels), dim=0).long()

            train_data = train_data.to(device)
            train_labels = train_labels.to(device)

            output = model(train_data)
            loss = criterion(output, train_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            train_loss += loss.item()
            train_predicted.extend(torch.argmax(output, dim=1).cpu().numpy())
            train_actual.extend(train_labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_actual, train_predicted)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # EVALUATION
        model.eval()
        test_loss = 0
        test_predicted, test_actual = [], []
        with torch.no_grad():
            for test_data, test_labels in test_loader:
                test_data = test_data.squeeze(1) if test_data.dim() == 4 else test_data
                test_data, test_labels = test_data.to(device).float(), test_labels.to(device).long()

                output = model(test_data)
                loss = criterion(output, test_labels)

                test_loss += loss.item()
                test_predicted.extend(torch.argmax(output, dim=1).cpu().numpy())
                test_actual.extend(test_labels.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc = accuracy_score(test_actual, test_predicted)
        test_kappa = cohen_kappa_score(test_actual, test_predicted)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        avg_acc += test_acc
        if scheduler:
            scheduler.step(test_loss)

        # Logging
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} Kappa: {test_kappa:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {time.time()-start_time:.1f}s")
        log_writer.write(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                         f"Test Loss {test_loss:.4f} Acc {test_acc:.4f} Kappa {test_kappa:.4f}\n")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_kappa = test_kappa
            best_model = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # GradCAM snapshot every 10 epochs
        if (epoch+1) % 10 == 0:
            try:
                sample_data, _ = next(iter(test_loader))
                sample_data = sample_data.squeeze(1).to(device).float()

                # Fallback if your model doesn't have layers[-1]
                target_layer = getattr(model, 'layers', None)
                if target_layer:
                    gradcam = GradCAM(model, target_layers=[target_layer[-1]], use_cuda=torch.cuda.is_available())
                    cam_map = gradcam(sample_data, target_category=0)[0]  # first sample
                    cam_map = cam_map.mean(axis=0)
                    cam_map = (cam_map - cam_map.min()) / (1e-7 + cam_map.max() - cam_map.min())
                    plot_gradcam_on_eeg(sample_data[0], cam_map, epoch+1, save_path, sample_idx=0)
            except Exception as e:
                print(f"GradCAM snapshot failed: {e}")

        # Early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    avg_acc /= (epoch+1)

    print("\nTraining complete.")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Best Kappa: {best_kappa:.4f}")
    log_writer.write(f"Best Accuracy: {best_acc:.4f} Kappa: {best_kappa:.4f}\n")
    log_writer.close()

    # Save best model
    torch.save(best_model, os.path.join(save_path, 'best_model.pth'))

    # Final classification report
    print("\nClassification Report on last test predictions:")
    print(classification_report(test_actual, test_predicted))
    print("Confusion Matrix:")
    print(confusion_matrix(test_actual, test_predicted))

    # Plot learning curves
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'learning_curves.png'))
    plt.close()

    return best_acc, best_kappa
