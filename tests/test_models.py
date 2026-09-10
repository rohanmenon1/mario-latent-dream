import torch
import torch.nn.functional as F

from mario_dreamer.models import (
    ConvDecoder,
    ConvDecoderConfig,
    ConvEncoder,
    ConvEncoderConfig,
    RSSM,
    RSSMConfig,
    RSSMState,
)


def test_conv_encoder_shape_and_gradients():
    encoder = ConvEncoder(ConvEncoderConfig(in_channels=3, embed_dim=128, depth=8))
    obs = torch.rand(5, 3, 64, 64)

    embed = encoder(obs)
    loss = embed.square().mean()
    loss.backward()

    assert embed.shape == (5, 128)
    assert torch.isfinite(embed).all()
    assert any(param.grad is not None for param in encoder.parameters())


def test_conv_decoder_shape_range_and_gradients():
    decoder = ConvDecoder(ConvDecoderConfig(out_channels=3, feature_dim=256, depth=8))
    features = torch.randn(5, 256)

    logits = decoder(features)
    recon = decoder.reconstruct(features)
    loss = logits.square().mean()
    loss.backward()

    assert logits.shape == (5, 3, 64, 64)
    assert recon.shape == (5, 3, 64, 64)
    assert torch.all((recon >= 0.0) & (recon <= 1.0))
    assert any(param.grad is not None for param in decoder.parameters())


def test_conv_encoder_decoder_support_larger_square_observation():
    encoder = ConvEncoder(ConvEncoderConfig(in_channels=3, obs_size=96, embed_dim=64, depth=4))
    decoder = ConvDecoder(ConvDecoderConfig(out_channels=3, obs_size=96, feature_dim=80, depth=4))
    obs = torch.rand(2, 3, 96, 96)
    features = torch.randn(2, 80)

    embed = encoder(obs)
    logits = decoder(features)

    assert embed.shape == (2, 64)
    assert logits.shape == (2, 3, 96, 96)


def test_rssm_observe_shapes():
    rssm = RSSM(
        RSSMConfig(
            action_dim=7,
            embed_dim=64,
            deter_dim=32,
            stoch_classes=4,
            stoch_categories=8,
            hidden_dim=64,
        )
    )
    embeds = torch.randn(3, 6, 64)
    action_indices = torch.randint(0, 7, (3, 6))
    actions = F.one_hot(action_indices, num_classes=7).float()

    posterior, prior = rssm.observe(embeds, actions)

    assert posterior.deter.shape == (3, 6, 32)
    assert posterior.stoch.shape == (3, 6, 4, 8)
    assert posterior.logits.shape == (3, 6, 4, 8)
    assert prior.deter.shape == (3, 6, 32)
    assert prior.stoch.shape == (3, 6, 4, 8)

    first_state = RSSMState(
        deter=posterior.deter[:, 0],
        stoch=posterior.stoch[:, 0],
        logits=posterior.logits[:, 0],
    )
    assert rssm.get_features(first_state).shape == (3, 64)


def test_rssm_observe_resets_state_and_previous_action_at_is_first():
    rssm = RSSM(
        RSSMConfig(
            action_dim=3,
            embed_dim=8,
            deter_dim=8,
            stoch_classes=2,
            stoch_categories=4,
            hidden_dim=8,
        )
    )
    embeds = torch.randn(1, 3, 8)
    actions = F.one_hot(torch.tensor([[1, 2, 1]]), num_classes=3).float()
    is_first = torch.tensor([[0.0, 1.0, 0.0]])
    captured = []
    original_obs_step = rssm.obs_step

    def capture_obs_step(previous, action, embed):
        captured.append((previous.deter.detach().clone(), action.detach().clone()))
        return original_obs_step(previous, action, embed)

    rssm.obs_step = capture_obs_step
    rssm.observe(embeds, actions, is_first=is_first)

    assert torch.count_nonzero(captured[1][0]) == 0
    assert torch.count_nonzero(captured[1][1]) == 0


def test_rssm_imagine_shapes_kl_and_gradients():
    rssm = RSSM(
        RSSMConfig(
            action_dim=7,
            embed_dim=64,
            deter_dim=32,
            stoch_classes=4,
            stoch_categories=8,
            hidden_dim=64,
        )
    )
    start = rssm.initial(batch_size=2)
    actions = F.one_hot(torch.randint(0, 7, (2, 5)), num_classes=7).float()

    prior = rssm.imagine(actions, start)
    posterior = RSSMState(
        deter=prior.deter,
        stoch=prior.stoch,
        logits=prior.logits + 0.1,
    )
    kl = rssm.kl_loss(posterior, prior)
    loss = prior.deter.square().mean() + kl.mean()
    loss.backward()

    assert prior.deter.shape == (2, 5, 32)
    assert prior.stoch.shape == (2, 5, 4, 8)
    assert kl.shape == (2, 5)
    assert torch.all(kl >= -1e-6)
    assert any(param.grad is not None for param in rssm.parameters())


def test_rssm_unimix_keeps_probabilities_nonzero():
    rssm = RSSM(
        RSSMConfig(
            action_dim=2,
            embed_dim=8,
            deter_dim=8,
            stoch_classes=2,
            stoch_categories=4,
            hidden_dim=8,
            unimix_ratio=0.01,
        )
    )
    logits = torch.tensor([[[1000.0, -1000.0, -1000.0, -1000.0]]])

    probs = rssm._probs(logits)

    assert torch.all(probs > 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))
