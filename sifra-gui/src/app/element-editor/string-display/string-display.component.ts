import { Component, Input, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'string-display',
    template: `
    <span class="string-display-container">
        <div class="string-display-name">
            {{name}}
        </div>
        <div class="string-display-body">
            <ndv-edit
                [title]="'value'"
                [placeholder]="_value"
                [permission]="true"
                (onSave)="doPublish($event);">
            </ndv-edit>
        </div>
    </span>
    `
})
export class StringDisplayComponent {
    @Input() name: string;
    @Input() numeric: boolean = false;
    @Input()
    set value(value: any) {
        this._value = value._value || value.default || value;
    }

    @Output() publish = new EventEmitter();

    _value: string;

    doPublish($event) {
        this._value = $event.value;
        this.publish.emit({
            name: this.name,
            value: this.numeric ? Number(this._value) : this._value
        });
    }
}

