import { SifraGuiPage } from './app.po';

describe('sifra-gui App', function() {
  let page: SifraGuiPage;

  beforeEach(() => {
    page = new SifraGuiPage();
  });

  it('should display message saying app works', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('app works!');
  });
});
