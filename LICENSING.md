# Licensing

- The existing onwemu source outside `psp/` remains under the MIT License in [`LICENSE`](LICENSE).
- The PSP workspace, including `psp-runtime`, is licensed under the GNU Affero General Public License version 3. Its full license text is [`psp/LICENSE`](psp/LICENSE), and its Cargo workspace metadata now says `AGPL-3.0-only`.
- The `web` feature links `psp-runtime` into the same WebAssembly program as onwemu. GNU's licensing guidance treats linked modules as a combined work, so the PSP-enabled browser build must be distributed under AGPL terms and is not MIT-only. This does not relicense the PSP source as MIT, or remove the MIT license from onwemu source used separately.

The browser footer links to this repository's source and license notices. If the combined program is conveyed, provide its corresponding source and preserve the applicable license notices as required by AGPLv3. See the [GNU AGPL](https://www.gnu.org/licenses/agpl.en.html) and [GNU guidance on combined works](https://www.gnu.org/licenses/gpl-faq.en.html#GPLStaticVsDynamic).
