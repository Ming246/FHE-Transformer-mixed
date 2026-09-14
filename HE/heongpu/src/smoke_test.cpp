#include <heongpu/heongpu.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr auto kScheme = heongpu::Scheme::CKKS;
constexpr double kScale = static_cast<double>(1ULL << 30);
constexpr double kTol = 0.05;

bool near(double expected, double actual, double tol = kTol)
{
    return std::abs(expected - actual) <= tol;
}

void check(bool ok, const char* label)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << std::endl;
    if (!ok)
    {
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main()
{
    std::cout << "HEonGPU smoke test (CKKS encode/decode/add/mul)\n";

    heongpu::HEContext<kScheme> context = heongpu::GenHEContext<kScheme>();
    const size_t poly_modulus_degree = 8192;
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    context->generate();

    heongpu::HEKeyGenerator<kScheme> keygen(context);
    heongpu::Secretkey<kScheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<kScheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<kScheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    heongpu::HEEncoder<kScheme> encoder(context);
    heongpu::HEEncryptor<kScheme> encryptor(context, public_key);
    heongpu::HEDecryptor<kScheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<kScheme> ops(context, encoder);

    const int slot_count = static_cast<int>(poly_modulus_degree / 2);
    std::vector<double> message(slot_count, 0.0);
    message[0] = 1.5;
    message[1] = -2.25;
    message[2] = 3.0;
    message[3] = 0.125;

    // --- encode / decode round-trip ---
    heongpu::Plaintext<kScheme> plain_in(context);
    encoder.encode(plain_in, message, kScale);

    std::vector<double> decoded;
    encoder.decode(decoded, plain_in);
    check(near(message[0], decoded[0]), "encode/decode slot 0");
    check(near(message[1], decoded[1]), "encode/decode slot 1");
    check(near(message[2], decoded[2]), "encode/decode slot 2");
    check(near(message[3], decoded[3]), "encode/decode slot 3");

    // --- homomorphic addition: (a + b) ---
    std::vector<double> lhs(slot_count, 0.0);
    std::vector<double> rhs(slot_count, 0.0);
    lhs[0] = 4.0;
    lhs[1] = -1.0;
    rhs[0] = 0.5;
    rhs[1] = 2.0;

    heongpu::Plaintext<kScheme> plain_lhs(context);
    heongpu::Plaintext<kScheme> plain_rhs(context);
    encoder.encode(plain_lhs, lhs, kScale);
    encoder.encode(plain_rhs, rhs, kScale);

    heongpu::Ciphertext<kScheme> ct_lhs(context);
    heongpu::Ciphertext<kScheme> ct_rhs(context);
    encryptor.encrypt(ct_lhs, plain_lhs);
    encryptor.encrypt(ct_rhs, plain_rhs);

    heongpu::Ciphertext<kScheme> ct_sum(context);
    ops.add(ct_lhs, ct_rhs, ct_sum);

    heongpu::Plaintext<kScheme> plain_sum(context);
    decryptor.decrypt(plain_sum, ct_sum);
    std::vector<double> sum_out;
    encoder.decode(sum_out, plain_sum);
    check(near(lhs[0] + rhs[0], sum_out[0]), "homomorphic add slot 0");
    check(near(lhs[1] + rhs[1], sum_out[1]), "homomorphic add slot 1");

    // --- homomorphic multiplication: (x * x), with relinearize + rescale ---
    std::vector<double> square_in(slot_count, 0.0);
    square_in[0] = 3.0;
    square_in[1] = -2.0;

    heongpu::Plaintext<kScheme> plain_sq(context);
    encoder.encode(plain_sq, square_in, kScale);

    heongpu::Ciphertext<kScheme> ct_sq(context);
    encryptor.encrypt(ct_sq, plain_sq);

    ops.multiply_inplace(ct_sq, ct_sq);
    ops.relinearize_inplace(ct_sq, relin_key);
    ops.rescale_inplace(ct_sq);

    heongpu::Plaintext<kScheme> plain_sq_out(context);
    decryptor.decrypt(plain_sq_out, ct_sq);
    std::vector<double> square_out;
    encoder.decode(square_out, plain_sq_out);
    check(near(square_in[0] * square_in[0], square_out[0]), "homomorphic mul slot 0");
    check(near(square_in[1] * square_in[1], square_out[1]), "homomorphic mul slot 1");

    std::cout << "All smoke checks passed.\n";
    return EXIT_SUCCESS;
}
